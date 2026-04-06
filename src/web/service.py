# 服务类
class TitleService:
    # 初始化
    def __init__(self, predictor):
        self.predictor = predictor

    # 核心服务：预测分类标签
    def predict(self, title):
        return self.predictor.predict(title)
