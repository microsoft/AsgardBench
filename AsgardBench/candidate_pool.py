class CandidatePool:
    def __init__(
        self, candidates=None, secondary_candidates=None, do_all_candidates=False
    ):
        self.candidates = candidates if candidates is not None else []
        self.secondary_candidates = (
            secondary_candidates if secondary_candidates is not None else []
        )
        self.do_all_candidates = do_all_candidates

        self.candidates_to_try = self.candidates.copy()
        self.secondary_candidates_to_try = self.secondary_candidates.copy()

    def next_candidate(self):

        if len(self.candidates_to_try) > 0:
            return self.candidates.pop(0)

        elif len(self.secondary_candidates_to_try) > 0:
            return self.secondary_candidates.pop(0)

        return None
