class LRWarmup():
    def __init__(self, epochs, max_lr, k):
        self.epochs = epochs
        self.max_lr = max_lr
        self.max_point = int(k * self.epochs)

    def __call__(self, epoch):
        self.lr_warmup(epoch)

    def lr_warmup(self, epoch):
        a_1 = self.max_lr / self.max_point
        a_2 = self.max_lr / (self.max_point - self.epochs)

        b = - a_2 * self.epochs

        return min(a_1 * epoch, a_2 * epoch + b)

