import sys
sys.path.insert(0, ".")
try:
    from alpha_expr._core import *
    print("Success: _core imported")
    # List available names
    import inspect
    names = [name for name in dir() if not name.startswith('_')]
    print(f"Exported names: {names[:20]}...")
except Exception as e:
    print(f"Error importing _core: {e}")
    import traceback
    traceback.print_exc()