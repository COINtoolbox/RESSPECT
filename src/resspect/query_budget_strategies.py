import numpy as np
from resspect.batch_functions import *

def batch_queries_uncertainty(class_probs, id_name, queryable_ids,
                              pool_metadata, budgets, criteria):
    """Select batch of queries based on acquistion criteria. Independently
    models the elements of the batch.

    Parameters
    ----------
    class_prob: np.array
        Classification probability. One value per class per object.
    id_name: str
        key to index ids from pool_metadata
    queryable_ids: np.array
        Set of ids for objects available for querying.
    pool_metadata: pandas Dataframe
        Contains infromation relevant to the poolset such as costs and ids.
    budgets: tuple of ints
        budgets for each of the telescopes assumes 0th index is for 4m and
        1th index is for 8m.
    criteria: str
        Acqution strategy to use can be 'uncertainty', 'entropy', 'margin',
        'least_confident' and 'random'.

    Returns
    -------
    acquistion_index: list
            List of indexes identifying the objects from the pool sampled to be
            queried. Guranteed to be within budget.
    """
    pool_ids = pool_metadata[id_name].values
    budget_4m = budgets[0]
    budget_8m = budgets[1]
    index_to_ids = {i: q_id for i, q_id in enumerate(queryable_ids)}
    ids_to_index = {i_d: i for i, i_d in enumerate(pool_metadata['id'].values)}
    pool_query_filter = np.array([p_id in queryable_ids for p_id in pool_ids])

    cost_4m = pool_metadata['cost_4m'].values[pool_query_filter]
    cost_8m = pool_metadata['cost_8m'].values[pool_query_filter]
    cost_4m[cost_4m >= 9999.0] = 1e8
    cost_8m[cost_8m >= 9999.0] = 1e8
    possible_4m = cost_4m < 1e8
    possible_8m = cost_8m < 1e8
    query_ids = pool_ids[pool_query_filter]

    class_probs = class_probs[pool_query_filter]

    if criteria == 'uncertainty':
        score = abs(class_probs[:, 1] - 0.5)
        reversed = False
    elif criteria == 'entropy':
        entropies = (-1*np.sum(class_probs * np.log(class_probs + 1e-12), axis=1))
        score = entropies
        reversed = True
    elif criteria == 'margin':
        sorted_probs = np.sort(class_probs, axis=1)
        score = sorted_probs[:, -1] - sorted_probs[:, -2]
        reversed = False
    elif criteria == 'least_confident':
        score = class_probs.max(axis=1)
        reversed = False
    elif criteria == 'random':
        score = np.random.rand(class_probs.shape[0])
        reversed = False

    cost_4m_possible = cost_4m[possible_4m]
    score_4m = score[possible_4m]
    if reversed:
        order_4m = score_4m.argsort()[::-1]
    else:
        order_4m = score_4m.argsort()

    cost_4m_order = cost_4m_possible[order_4m]
    query_ids_4m_possible = query_ids[possible_4m][order_4m]

    # Record acquistions as IDs
    acquistions_4m = []
    total_cost_4m = 0.
    for q_id, c_4m in zip(query_ids_4m_possible, cost_4m_order):
        if (total_cost_4m + c_4m) < budget_4m:
            acquistions_4m.append(q_id)
            total_cost_4m += c_4m

    acquistions =  acquistions_4m.copy()

    cost_8m_possible = cost_8m[possible_8m]
    score_8m = score[possible_8m]
    if reversed:
        order_8m = score_8m.argsort()[::-1]
    else:
        order_8m = score_8m.argsort()

    cost_8m_order = cost_8m_possible[order_8m]
    query_ids_8m_possible = query_ids[possible_8m][order_8m]

    acquistions_8m = []
    total_cost_8m = 0.
    for q_id, c_8m in zip(query_ids_8m_possible, cost_8m_order):
        if (total_cost_8m + c_8m) < budget_8m and (q_id not in acquistions):
            acquistions_8m.append(q_id)
            total_cost_8m += c_8m

    acquistions += acquistions_8m

    if total_cost_4m > budget_4m:
        raise RuntimeError("4m Budget exceeded")
    if total_cost_8m > budget_8m:
        raise RuntimeError("8m Budget exceeded")
    if len(acquistions) != len(set(acquistions)):
        raise RuntimeError("Some acquistions were repeated")
    if len(set(acquistions_4m) & set(acquistions_8m)) != 0:
        raise RuntimeError("Object acquired by both telescopes")

    acquistion_index = []
    for q_id in acquistions:
        acquistion_index.append(ids_to_index[q_id])

    return acquistion_index

def batch_queries_mi_entropy(probs_B_K_C, id_name, queryable_ids,
                             pool_metadata, budgets, criteria="MI" ):
    """Select batch of queries based on acquistion criteria. Jointly models the
    elements of the batch.

    Parameters
    ----------
    probs_B_K_C: np.array
        Classification probabilitity distributions for each datapoint for each
        model in the committee (or sample from model posterior). B is the number
        of data points, K is the committee size and C is the number of classes.
    id_name: str
        key to index ids from pool_metadata
    queryable_ids: np.array
        Set of ids for objects available for querying.
    pool_metadata: pandas Dataframe
        Contains infromation relevant to the poolset such as costs and ids.
    budgets: tuple of ints
        budgets for each of the telescopes assumes 0th index is for 4m and
        1th index is for 8m.
    criteria: str
        Acqution strategy to use can be 'uncertainty', 'entropy', 'margin',
        'least_confident' and 'random'.

    Returns
    -------
    acquistion_index: list
            List of indexes identifying the objects from the pool sampled to be
            queried. Guranteed to be within budget.
    """
    pool_ids = pool_metadata[id_name].values
    # Specifically queryable ids since we don't need ids to the pool in general.
    index_to_ids = {i: q_id for i, q_id in enumerate(queryable_ids)}
    ids_to_index = {i_d: i for i, i_d in enumerate(pool_metadata['id'].values)}
    pool_query_filter = np.array([p_id in queryable_ids for p_id in pool_ids])

    cost_4m = pool_metadata['cost_4m'].values[pool_query_filter]
    cost_8m = pool_metadata['cost_8m'].values[pool_query_filter]
    cost_4m[cost_4m >= 9999.0] = np.inf
    cost_8m[cost_8m >= 9999.0] = np.inf

    # For numerical reasons ie divide by zero etc.
    probs_B_K_C = probs_B_K_C[pool_query_filter]
    probs_B_K_C += 1.e-12

    conditional_entropies_B = compute_conditional_entropies_B(probs_B_K_C)
    B, K, C = probs_B_K_C.shape
    num_samples_per_ws = 40000 // K
    num_samples = num_samples_per_ws * K

    budget_4m = budgets[0]
    budget_8m = budgets[1]
    acquistions = []
    acquistions_4m = []
    acquistions_8m = []
    total_cost_4m = 0.
    total_cost_8m = 0.
    scores = []
    prev_joint_probs_M_K = None
    prev_samples_M_K = None
    top_scores = []

    is_time = True
    i = 0
    while is_time:
        #print(i)
        exact_samples = C ** i
        if exact_samples <= num_samples:
            if len(acquistions) == 0:
                joint_entropies_B = exact_batch(probs_B_K_C)
            else:
                prev_joint_probs_M_K = joint_probs_M_K(probs_B_K_C[acquistions[-1][None]], prev_joint_probs_M_K)
                joint_entropies_B = exact_batch(probs_B_K_C, prev_joint_probs_M_K)
        else:
            # Clear memory will be using sampling method from here on out.
            prev_joint_probs_M_K = None
            prev_samples_M_K = sample_M_K(probs_B_K_C[acquistions], S=num_samples_per_ws)
            joint_entropies_B = batch_sample(probs_B_K_C, prev_samples_M_K)

        if criteria == 'MI':
            batch_scores = joint_entropies_B - conditional_entropies_B
            #print(batch_scores.max())
            batch_scores = batch_scores - np.sum(conditional_entropies_B[acquistions])
            #print(batch_scores.max())
        elif criteria == 'entropy':
            batch_scores = joint_entropies_B

        # Adjust scores for cost
        scores_4m = batch_scores / cost_4m
        scores_4m[~np.isfinite(scores_4m)] = -np.inf
        scores_8m = batch_scores / cost_8m
        scores_8m[~np.isfinite(scores_8m)] = -np.inf

        scores_4m[acquistions] = -1 * np.inf
        scores_8m[acquistions] = -1 * np.inf

        # What objects can be observered within budget
        possible_4m = (cost_4m + total_cost_4m) <= budget_4m
        possible_4m[~np.isfinite(scores_4m)] = False
        possible_8m = (cost_8m + total_cost_8m) <= budget_8m
        possible_8m[~np.isfinite(scores_8m)] = False

        sorted_4m_idx = scores_4m.argsort()[::-1]
        possible_4m_order = np.where(possible_4m[sorted_4m_idx])[0]

        sorted_8m_idx = scores_8m.argsort()[::-1]
        possible_8m_order = np.where(possible_8m[sorted_8m_idx])[0]

        if np.any(possible_4m) and np.any(possible_8m):
            #print("BOTH POSSIBLE")
            top_4m_score = scores_4m[sorted_4m_idx[possible_4m_order]][0]
            top_8m_score = scores_8m[sorted_8m_idx[possible_8m_order]][0]
            if top_4m_score >= top_8m_score:
                #print("Choose 4m")
                top_score = top_4m_score
                selection = sorted_4m_idx[possible_4m_order][0]
                acquistions_4m.append(selection)
                total_cost_4m += cost_4m[selection]
            else:
                #print("Choose 8m")
                top_score = top_8m_score
                selection = sorted_8m_idx[possible_8m_order][0]
                acquistions_8m.append(selection)
                total_cost_8m += cost_8m[selection]

        elif np.any(possible_4m) and not np.any(possible_8m):
            #print("Only 4m possible")
            top_4m_score = scores_4m[sorted_4m_idx[possible_4m_order]][0]
            top_score = top_4m_score
            selection = sorted_4m_idx[possible_4m_order][0]
            acquistions_4m.append(selection)
            total_cost_4m += cost_4m[selection]

        elif not np.any(possible_4m) and np.any(possible_8m):
            #print("Only 8m possible")
            top_8m_score = scores_8m[sorted_8m_idx[possible_8m_order]][0]
            top_score = top_8m_score
            selection = sorted_8m_idx[possible_8m_order][0]
            acquistions_8m.append(selection)
            total_cost_8m += cost_8m[selection]

        elif not np.any(possible_4m) and not np.any(possible_8m):
            #print("Budget Full")
            is_time = False
            continue

        acquistions.append(selection)
        scores.append(batch_scores[selection])
        i += 1
        top_scores.append(top_score)
        #print("TOP SCORE: {}".format(top_score))
        #print(acquistions[-1], total_cost_4m, total_cost_8m)
        #print()

    if total_cost_4m > budget_4m:
        raise RuntimeError("4m Budget exceeded")
    if total_cost_8m > budget_8m:
        raise RuntimeError("8m Budget exceeded")
    if len(acquistions) != len(set(acquistions)):
        raise RuntimeError("Some acquistions were repeated")
    if len(set(acquistions_4m) & set(acquistions_8m)) != 0:
        raise RuntimeError("Object acquired by both telescopes")

    acquistion_ids = []
    for index in acquistions:
        acquistion_ids.append(index_to_ids[index])

    acquistion_indexes = []
    for p_id in acquistion_ids:
        acquistion_indexes.append(ids_to_index[p_id])

    return acquistion_indexes
