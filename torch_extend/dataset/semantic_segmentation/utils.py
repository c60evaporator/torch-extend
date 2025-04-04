from abc import abstractmethod

class SemanticSegOutput():
    @abstractmethod
    def get_image_target_path(self, index: int):
        """Get the image and target path of the dataset."""
        raise NotImplementedError
