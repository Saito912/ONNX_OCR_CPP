# ctc_decode_py.py
import numpy as np
import math


# 用于合并 log 概率的辅助函数 (log-sum-exp trick)
def log_add(a, b):
    """
    在对数空间中安全地计算 a + b 的和。
    log(exp(a) + exp(b))
    """
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))


def ctc_beam_search_decode_py(logits: np.ndarray, beam_size: int = 10) -> list[str]:
    """
    纯 Python 实现的 CTC Beam Search 解码器。

    Args:
        logits (np.ndarray): 模型输出的 logits 张量，shape 为 (B, T, C)。
        beam_size (int): beam search 的宽度。

    Returns:
        list[str]: 解码后的字符串列表。
    """
    charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    blank_id = 0

    B, T, C = logits.shape

    if C != len(charset) + 1:
        raise ValueError(f"无效的类别数。期望 {len(charset) + 1}, 得到 {C}")

    results = []

    # 对 batch 中的每个样本进行处理
    for b in range(B):
        batch_logits = logits[b]  # Shape: (T, C)

        # 1. 计算 Log Softmax
        # 为了数值稳定性，先减去最大值
        max_val = np.max(batch_logits, axis=1, keepdims=True)
        exp_logits = np.exp(batch_logits - max_val)
        sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
        log_softmax = batch_logits - max_val - np.log(sum_exp)

        # 初始化 beam
        # beam 中的元素是 (prefix, log_prob)
        # prefix 使用 tuple 以便作为字典的 key
        beam = [(tuple(), 0.0)]  # ((), log(1))

        # 2. 遍历所有时间步
        for t in range(T):
            log_probs_t = log_softmax[t]  # 当前时间步的 log 概率

            # 使用字典来合并具有相同前缀的路径
            candidates = {}

            # 3. 扩展当前 beam 中的每个假设
            for prefix, prev_log_prob in beam:
                # 遍历所有类别（包括 blank）
                for c in range(C):
                    log_prob_c = log_probs_t[c]
                    new_log_prob = prev_log_prob + log_prob_c

                    if c == blank_id:
                        # 情况 1: 输出 blank
                        # 前缀不变，概率更新
                        candidates[prefix] = log_add(candidates.get(prefix, -np.inf), new_log_prob)
                    else:
                        # 情况 2: 输出一个字符
                        new_prefix = prefix
                        # 如果新字符与前缀的最后一个字符不同，则添加
                        if not prefix or c != prefix[-1]:
                            new_prefix = prefix + (c,)

                        candidates[new_prefix] = log_add(candidates.get(new_prefix, -np.inf), new_log_prob)

            # 4. 剪枝：排序并保留 top-k
            # 将 candidates 字典转为列表
            sorted_beam = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
            beam = sorted_beam[:beam_size]

        # 5. 解码最终结果
        best_prefix, _ = beam[0]
        decoded_str = "".join([charset[label_id - 1] for label_id in best_prefix])
        results.append(decoded_str)

    return results