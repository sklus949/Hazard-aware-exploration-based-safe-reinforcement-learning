

class XY:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Predict:
    def __init__(self, x, y, vx, vy):
        self.N = 5
        self.T = 0.25
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.pred_list = []

    def predict(self):
        pred1 = XY(0, 0)
        pred1.x = [self.x.item()]
        pred1.y = [self.y.item()]
        vx = self.vx.item()
        vy = self.vy.item()
        x = self.x.item()
        y = self.y.item()
        self.pred_list.append(pred1)
        for i in range(self.N):
            x1 = x + vx * self.T
            y1 = y + vy * self.T
            x = x1
            y = y1
            pred = XY(0, 0)
            pred.x = [x]
            pred.y = [y]
            self.pred_list.append(pred)
