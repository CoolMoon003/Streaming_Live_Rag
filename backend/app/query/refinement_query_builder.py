class RefinementQueryBuilder:

    REPLACEMENT_PREFIXES = [
        "no, i meant ",
        "no i meant ",
        "i meant ",
        "actually ",
        "instead, ",
        "instead ",
        "not that, ",
        "not that ",
        "forget that, ",
        "forget that ",
    ]

    ADDITIVE_PREFIXES = [
        "and what about ",
        "and also ",
        "also ",
        "what about ",
        "additionally ",
        "in addition ",
    ]

    def build(
        self,
        previous_query: str,
        refinement_query: str,
        refinement_type: str = "ADDITIVE",
    ) -> dict:

        previous = previous_query.strip()
        refinement = refinement_query.strip()

        lowered = refinement.lower()

        if refinement_type == "REPLACEMENT":

            cleaned = refinement

            for prefix in self.REPLACEMENT_PREFIXES:

                if lowered.startswith(prefix):

                    cleaned = refinement[
                        len(prefix):
                    ].strip()

                    break

            return {
                "query": cleaned.rstrip(".?!"),
                "method": "replacement_cleanup",
            }

        if not previous:

            return {
                "query": refinement,
                "method": "direct",
            }

        cleaned = refinement

        for prefix in self.ADDITIVE_PREFIXES:

            if lowered.startswith(prefix):

                cleaned = refinement[
                    len(prefix):
                ].strip()

                break

        query = (
            f"{previous.rstrip('?')} "
            f"regarding {cleaned.rstrip('?')}"
        )

        return {
            "query": query,
            "method": "contextual_merge",
        }