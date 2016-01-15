"""Appender class"""
class Appender(object):
    """
    This class is derived from matplotlib.docstring (1.1.0) module
    http://matplotlib.sourceforge.net/users/license.html

    This function appends an addendum to the docstring of the
    target function.
    """
    def __init__(self, addendum, join='', indents=0):
        def indent(text, indents=1):
            """Indent text"""
            if not text or not isinstance(text, str):
                return ''
            jointext = ''.join(['\n'] + ['    '] * indents)
            return jointext.join(text.split('\n'))
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
