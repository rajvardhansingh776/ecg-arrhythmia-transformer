class EarlyStopping:

    def __init__(self,patience=6):

        self.patience=patience

        self.best=None

        self.count=0

    def step(self,val):

        if self.best is None or val<self.best:

            self.best=val

            self.count=0

            return False

        self.count+=1

        return self.count>=self.patience