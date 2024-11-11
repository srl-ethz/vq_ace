# robomimic simulator dependens on robomimic

try:
    import robomimic
    from .env_runner import RobomimicEnv
except ImportError:
    print("Warning: robomimic not installed, robomimic simulator will not be available.")
    pass
