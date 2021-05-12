class a:
    def __call__ (self, t):
        return 1+t

class b(a):
    def __init__ (self):
        super(b, self).__init__()

    def __call__ (self, t):
        return 1 + super(b, self).__call__(t)

if __name__ == "__main__":
    t = a()
    print(t(7))
    c = b()
    print(c(8))
