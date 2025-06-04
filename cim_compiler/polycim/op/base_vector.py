class Base:
    def __init__(self, corrdinate, reuse_array_id, is_trival):
        self.corrdinate = tuple(corrdinate)
        self.reuse_array_id = reuse_array_id
        self.is_trival = is_trival
        self.n_non_zero = sum([int(i != 0) for i in corrdinate])
        self.is_skewed = self.n_non_zero >= 2

    def __str__(self):
        return f"Base(corrdinate={self.corrdinate}, reuse_array_id={self.reuse_array_id}, is_trival={self.is_trival})"

    def __eq__(self, other):
        if isinstance(other, Base):
            return self.corrdinate == other.corrdinate
        return False

    def __hash__(self):
        return hash(self.corrdinate)
