from models.base_model import BaseModel


class CO2Model(BaseModel):
    def __init__(self, data_path):
        super().__init__(data_path, 'CO2_Release')
