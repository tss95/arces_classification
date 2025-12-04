import sys
sys.modules['tensorflow'] = None
try:
    import gbf_iter_torch
    print('Import successful')
except ImportError as e:
    print(f'Import failed: {e}')
except Exception as e:
    print(f'Runtime error (expected if running script logic): {e}')

