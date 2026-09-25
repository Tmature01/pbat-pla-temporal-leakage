from models.base_model import BaseModel


class StrengthModel(BaseModel):
    def __init__(self, data_path):
        super().__init__(data_path, 'Tensile_Strength')
