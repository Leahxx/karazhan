#! /usr/bin/env python
# -*- coding UTF-8 -*-

"""
@Author : tangxx11
@Since  : 2024/10/24 14:51
"""

def validate(min_s: str, target: str):
    if len(min_s) < len(target):
        return False
    for item in target:
        if item not in min_s:
            return False
    return True


def minWindow(s: str, t: str):
    res = ""
    left = 0
    min_len = 1e10
    for right in range(len(s)):
        min_str = s[left:right+1]  # DANOCMB
        while validate(min_str, t):
            print(f"min str {right}: ", min_str)
            if len(min_str) < min_len:
                res = min_str
                min_len = len(min_str)
            min_str = s[left: right+1]  # DANOCMB ANOCMB
            left += 1
        print(f"res {right}: ", res)
    return res



if __name__ == '__main__':
    # s = "DANOCMBOBANC"
    # t = "XN"
    s = "bbaa"
    t = "aba"
    answer = "baa"
    answer = minWindow(s, t)
    print(answer)
