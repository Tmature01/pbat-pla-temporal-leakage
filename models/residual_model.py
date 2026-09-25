from models.base_model import BaseModel


class ResidualModel(BaseModel):
    def __init__(self, data_path):
        super().__init__(data_path, 'Residual_Rate')
