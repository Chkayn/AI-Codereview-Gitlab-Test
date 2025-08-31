import ast
import re
from typing import List, Set


def _extract_python_imports(code: str) -> List[str]:
    """
    使用 Python AST 抽取 import/from import 的依赖名。
    """
    deps: Set[str] = set()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name:
                        deps.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    deps.add(node.module.split('.')[0])
    except Exception:
        # 语法错误等，交给通用解析
        pass
    return list(deps)


def _extract_generic_deps_by_regex(code: str) -> List[str]:
    """
    针对非 Python 代码或 AST 失败的回退方案：
    - JS/TS: import xxx from 'pkg'; const x = require('pkg')
    - Go: import "pkg/xxx"
    - Java: import com.xxx.yyy;
    - C/C++: #include <xxx> 或 "xxx"
    - PHP: use Vendor\\Package;
    """
    deps: Set[str] = set()

    patterns = [
        r"import\s+[\w\{\}\*,\s]+\s+from\s+['\"]([^'\"\s]+)['\"]",   # ES import from
        r"import\s+['\"]([^'\"\s]+)['\"]",                          # ES bare import
        r"require\(\s*['\"]([^'\"\s]+)['\"]\s*\)",                  # CommonJS require
        r"import\s+([a-zA-Z0-9_\.]+);",                             # Java import
        r"#include\s*[<\"]([^>\"]+)[>\"]",                          # C/C++ include
        r"use\s+([A-Za-z0-9_\\]+);",                                # PHP use
        r"import\s+\"([^\"\s]+)\"",                                 # Go import "pkg"
    ]

    for p in patterns:
        for m in re.findall(p, code):
            # 取第一段作为主包名
            pkg = str(m).split('/')[0].split('.')[0]
            if pkg:
                deps.add(pkg)

    # 兜底：也尝试抓 Python 的 import/from import
    py_like = [
        r"from\s+([a-zA-Z0-9_\.]+)\s+import\s+",
        r"import\s+([a-zA-Z0-9_\.]+)",
    ]
    for p in py_like:
        for m in re.findall(p, code):
            pkg = str(m).split('.')[0]
            if pkg:
                deps.add(pkg)

    return list(deps)


def extract_dependencies_from_code(code: str) -> List[str]:
    """
    高优先级用 Python AST（能解析就拿），否则使用通用正则回退。
    """
    deps = _extract_python_imports(code)
    if deps:
        return deps
    return _extract_generic_deps_by_regex(code)