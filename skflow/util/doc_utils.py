class Appender(object):
    """
    This class is derived from matplotlib.docstring (1.1.0) module
    http://matplotlib.sourceforge.net/users/license.html

    A function decorator that will append an addendum to the docstring
    of the target function.
    This decorator should be robust even if func.__doc__ is None
    (for example, if -OO was passed to the interpreter).
    Usage: construct a docstring.Appender with a string to be joined to
    the original docstring. An optional 'join' parameter may be supplied
    which will be used to join the docstring and addendum. e.g.
    add_copyright = Appender("Copyright (c) 2009", join='\n')
    @add_copyright
    def my_dog(has='fleas'):
        "This docstring will have a copyright below"
        pass
    """
    def __init__(self, addendum, join='', indents=0):
        if indents > 0:
            self.addendum = indent(addendum, indents=indents)
        else:
            self.addendum = addendum
        self.join = join

    def __call__(self, func):
        func.__doc__ = func.__doc__ if func.__doc__ else ''
        self.addendum = self.addendum if self.addendum else ''
        docitems = [func.__doc__, self.addendum]
        func.__doc__ = self.join.join(docitems)
        return func
