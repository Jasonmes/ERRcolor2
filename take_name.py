# encoding: utf-8
import inspect
import re


# 拿到变量的名字
def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)',
                      line)
        if m:
            return m.group(1)
