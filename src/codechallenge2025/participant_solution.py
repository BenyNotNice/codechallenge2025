# src/codechallenge2025/participant_solution.py
"""
Easy Participant Template for #codechallenge2025

You ONLY need to implement the function: match_single

The find_matches function is provided for you — no need to change it!
"""

import pandas as pd
from typing import List, Dict, Any


def match_single(
    query_profile: Dict[str, Any], database_df: pd.DataFrame
) -> List[Dict]:
    """
    Find the top 10 candidate matches for a SINGLE query profile.

    Args:
        query_profile: dict with 'PersonID' and locus columns (e.g. {'PersonID': 'Q001', 'TH01': '9,9.3', ...})
        database_df: Full database as pandas DataFrame (500k rows)

    Returns:
        List of up to 10 candidate dicts, sorted by strength (best first):
        [
            {
                "person_id": "P000123",
                "clr": 1e15,                    # Combined Likelihood Ratio
                "posterior": 0.99999,           # Optional: posterior probability
                "consistent_loci": 20,
                "mutated_loci": 1,
                "inconclusive_loci": 0
            },
            ...
        ]
    """
    query_id = query_profile['PersonID']
    loci = [c for c in query_profile.keys() if c != 'PersonID']
    candidates = []

    for _, candidate_row in database_df.iterrows():
        cand_id = candidate_row['PersonID']
        if cand_id == query_id:
            continue

        clr = 1.0
        consistent = 0
        mutated = 0
        inconclusive = 0
        exclusions = 0
        identity_matches = 0
        compared = 0

        for locus in loci:
            # Parse alleles
            q_val = str(query_profile.get(locus, '-')).strip()
            c_val = str(candidate_row.get(locus, '-')).strip()

            if q_val in ('-', '') or c_val in ('-', ''):
                inconclusive += 1
                continue

            q_alleles = set(map(float, q_val.split(','))) if ',' in q_val else {float(q_val)}
            c_alleles = set(map(float, c_val.split(','))) if ',' in c_val else {float(c_val)}

            compared += 1
            if q_alleles == c_alleles:
                identity_matches += 1

            shared = q_alleles & c_alleles

            if shared:
                # Direct match - use simple scoring
                consistent += 1
                # Assume allele frequency ~0.15 (average), transmission prob 0.5
                lr = (1.0 if len(c_alleles) == 1 else 0.5) / 0.15
                clr *= lr
            elif any(abs(qa - ca) <= 1.0 for qa in q_alleles for ca in c_alleles if 0 < abs(qa - ca) <= 1.0):
                # Mutation match
                mutated += 1
                clr *= 0.002 / 0.15
            elif len(q_alleles) == 1 and len(c_alleles) == 1:
                # Both single allele, possible dropout
                inconclusive += 1
                clr *= 0.5
            else:
                # Exclusion
                exclusions += 1
                clr *= 0.01

        # Filter out bad matches
        if exclusions > 4 or consistent < 5:
            continue

        # Filter same-person (>80% identical)
        if compared > 0 and identity_matches / compared > 0.80:
            continue

        posterior = clr / (clr + 1.0) if clr > 0 else 0.0

        candidates.append({
            "person_id": cand_id,
            "clr": clr,
            "posterior": posterior,
            "consistent_loci": consistent,
            "mutated_loci": mutated,
            "inconclusive_loci": inconclusive
        })

    candidates.sort(key=lambda x: -x['clr'])
    return candidates[:10]


# ============================================================
# DO NOT MODIFY BELOW THIS LINE — This runs your function!
# ============================================================


def find_matches(database_path: str, queries_path: str) -> List[Dict]:
    """
    Main entry point — automatically tested by CI.
    Loads data and calls your match_single for each query.
    """
    print("Loading database and queries...")
    database_df = pd.read_csv(database_path)
    queries_df = pd.read_csv(queries_path)

    results = []

    print(f"Processing {len(queries_df)} queries...")
    for _, query_row in queries_df.iterrows():
        query_id = query_row["PersonID"]
        query_profile = query_row.to_dict()

        print(f"  Matching query {query_id}...")
        top_candidates = match_single(query_profile, database_df)

        results.append(
            {
                "query_id": query_id,
                "top_candidates": top_candidates[:10],  # Ensure max 10
            }
        )

    print("All queries processed.")
    return results
