"""
The class that inherits from AutoInit can register a name of cfg parameters and the function to execute at initialization.
Example:
class Algo(AutoInit, init_cfgname_and_funcs=[("algo_cfg". "init_algo")]):
    def init_algo(self, algo_cfg):
        # some initialization code
        ...
class TrainMixin(AutoInit, init_cfgname_and_funcs=[("train_cfg", "init_train")]):
    def init_train(self, train_cfg):
        # some initialization code
        ...
class Foo(Algo, TrainMixin):
    pass
Then the Foo class expects the "algo_cfg" and "train_cfg" parameters at initialization.
foo = Foo(algo_cfg=algo_cfg, train_cfg=train_cfg)
"""
class AutoInit:
    init_cfgname_and_funcs = ()
    def __init_subclass__(cls, cfgname_and_funcs=[], **kwargs): 
        super().__init_subclass__(**kwargs)
        cls.init_cfgname_and_funcs = ()
        for b_cls in cls.__bases__:
            if not hasattr(b_cls, "init_cfgname_and_funcs"):
                continue
            cls.init_cfgname_and_funcs += b_cls.init_cfgname_and_funcs

        init_funcnames = [fname for cfgname, fname in cls.init_cfgname_and_funcs]
        init_cfgnames = [cfgname for cfgname, fname in cls.init_cfgname_and_funcs]
        for cfgname, funcname in cfgname_and_funcs:
            func = getattr(cls, funcname)
            if not callable(func):
                raise TypeError(f"{funcname} must be a callable method")
            if cfgname in init_cfgnames:
                raise ValueError(f"{cfgname} is already in init_cfgname_and_funcs")
            if funcname in init_funcnames:
                raise ValueError(f"{funcname} is already in init_cfgname_and_funcs")
        cls.init_cfgname_and_funcs+=tuple(cfgname_and_funcs)

            
    def __init__(self, **kwargs):
        for cfgname, funcname in self.init_cfgname_and_funcs:
            func = getattr(self, funcname)
            if cfgname is not None and cfgname not in kwargs:
                raise ValueError(f"{cfgname} is not in the kwargs")
            cfg = kwargs[cfgname] if cfgname is not None else kwargs
            func(**(cfg))

    @classmethod
    def print_init_cfg_parameters(cls):
        print("The cfg parameters for", cls.__name__)
        for cfgname, _ in cls.init_cfgname_and_funcs:
            print(cfgname)
