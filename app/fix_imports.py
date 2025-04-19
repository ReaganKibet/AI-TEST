# fix_imports.py
class Resource:
    """Placeholder for missing openfabric_pysdk.fields.Resource"""
    pass

class SchemaUtil:
    """Placeholder for missing openfabric_pysdk.utility.SchemaUtil"""
    @staticmethod
    def create(obj, data):
        """Simple implementation to copy data to object"""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
        return obj