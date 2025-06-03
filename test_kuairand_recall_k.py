import sys
import numpy as np
from collections import defaultdict


def load_embeddings(recall_index_path):
    embeddings = []
    with open(recall_index_path, 'rb') as f:
        while True:
            try:
                emb = np.load(f)
                embeddings.append(emb)
            except ValueError:
                break
    return np.concatenate(embeddings, axis=0)


def summary_label(sample_label):

    # Sample_label: [user_id, item_id, is_like, is_follow]
    print(sample_label.shape)

    # Sample_id(user_id/item_id)
    sample_id = sample_label[:, [0, 1]].astype(int)
    label = sample_label[:, [2, 3]].astype(int)

    # SampleID
    print("First 5 sample_ids:")
    print(sample_id[:5])

    # Label
    print("\nFirst 5 embedding rows:")
    print(label[:5])


def check_consistency_label(sample_label):

    sample_label = sample_label.astype(int)
    # pair <user_id, item_id> for label consistency check
    user_item_pairs = sample_label[:, [0, 1]]

    unique_pairs, counts = np.unique(
        user_item_pairs, axis=0, return_counts=True)
    duplicate_pairs = unique_pairs[counts > 1]

    if len(duplicate_pairs) == 0:
        print("No duplicate pairs found.")
        return

    print(f"Found {len(duplicate_pairs)} duplicate pairs.")

    # check label consistency
    for pair in duplicate_pairs:
        indices = np.where((user_item_pairs[:, 0] == pair[0]) & (
            user_item_pairs[:, 1] == pair[1]))[0]

        labels = sample_label[indices, 2:]  # get label

        if not np.all(np.array_equal(labels[0], label) for label in labels):
            print(f"\nInconsistent labels found for duplicate pair: {pair}")
            print("Matching rows:")
            print(sample_label[indices])


def summary_data(total_embedding):

    # Total Embedding[Duplicate SampleID]
    print("Total Shape: ", total_embedding.shape)

    # Sample_id(user_id/item_id)
    sample_id = total_embedding[:, 0].astype(int)
    embedding_values = total_embedding[:, 1:]

    # SampleID
    print("First 5 sample_ids:")
    print(sample_id[:5])
    print("Unique Cnt: ", len(np.unique(sample_id)))     # Unique Cnt

    # Embedding
    print("\nFirst 5 embedding rows:")
    print(embedding_values[:5])


def check_consistency_data(total_embedding):
    # 0. Parser SampleId[item_id/user_id] & Embedding
    sample_id = total_embedding[:, 0].astype(int)
    embedding_values = total_embedding[:, 1:]

    # 1. Duplicated SampleId
    unique_ids, counts = np.unique(sample_id, return_counts=True)
    duplicate_ids = unique_ids[counts > 1]
    print(f"Found {len(duplicate_ids)} duplicate sample_ids.")

    # 2. Check Embedding Consistency
    inconsistency_flag = False
    for dup_id in duplicate_ids:
        indices = np.where(sample_id == dup_id)[0]
        dup_embeddings = embedding_values[indices]

        if not np.all(np.array_equal(dup_embeddings[0], emb) for emb in dup_embeddings):
            inconsistency_flag = True
            break

    if inconsistency_flag:
        print("[WARNING]duplicate sample_ids embedding [In-Consistency...]")
    else:
        print("[INFO]duplicate sample_ids have consistent embeddings.")


def deduplicated_data(total_embedding):
    # SampleId(User_id or Item_id) & Embedding
    sample_id = total_embedding[:, 0].astype(int)
    embedding_values = total_embedding[:, 1:]

    # Deduplicate
    unique_embeddings = {}
    for idx, s_id in enumerate(sample_id):
        unique_embeddings[s_id] = embedding_values[idx]

    # Sample_id (user_id/item_id)
    unique_sample_ids = np.array(list(unique_embeddings.keys()))
    # Embedding
    unique_embedding_values = np.array(list(unique_embeddings.values()))

    print(unique_sample_ids.shape, unique_sample_ids[:5])
    print(unique_embedding_values.shape, unique_embedding_values[:5])

    return unique_sample_ids, unique_embedding_values


def gen_label_dict(sample_label, reverse=False):
    # [user_id, item_id, is_like, is_follow]
    sample_label = sample_label.astype(int)

    # sample_id [user_id, item_id]
    sample_ids = sample_label[:, [0, 1]]

    is_like_dict = defaultdict(set)
    is_follow_dict = defaultdict(set)

    for row in sample_label:
        user_id, item_id, is_like_label, is_follow_label = row

        if is_like_label == 1:
            if reverse == False:
                is_like_dict[user_id].add(item_id)
            else:
                assert reverse == True
                is_follow_dict[user_id].add(item_id)

        if is_follow_label == 1:
            if reverse == False:
                is_follow_dict[user_id].add(item_id)
            else:
                assert reverse == True
                is_like_dict[user_id].add(item_id)

    return sample_ids, is_like_dict, is_follow_dict


def recall_k(unique_user_id, unique_user_embedding, unique_item_id, unique_item_embedding, is_like_dict, is_follow_dict, K=10):
    """
    Calculate the recall@K for each user.

    Args:
    - unique_user_id: all user_id.
    - unique_user_embedding: embedding of each user_id. (shape: [num_users, embedding_dim])
    - unique_item_id: all item_id.
    - unique_item_embedding: embedding of each item_id. (shape: [num_items, embedding_dim])
    - is_like_dict: A dict containing positive sample interactions (is_like). {user_id: {item_id_set}}
    - is_follow_dict:  A dict containing positive sample interactions (is_follow). {user_id: {item_id_set}}
    - K: int (default: 10)

    Return:
    - is_like_recall_at_k
    - is_follow_recall_at_k
    """
    recall_is_like_at_k = {}
    recall_is_follow_at_k = {}

    item_embedding_matrix = np.array(unique_item_embedding)
    item_ids = np.array(unique_item_id)

    for user_idx, user_id in enumerate(unique_user_id):
        # user embedding
        user_embedding = np.array(unique_user_embedding[user_idx])

        similarities = np.dot(item_embedding_matrix, user_embedding)

        sorted_indices = np.argsort(-similarities)
        top_k_indices = sorted_indices[:K]
        top_k_items = item_ids[top_k_indices]

        # is_like per user_id
        positive_items_like = is_like_dict.get(user_id, set())
        if len(positive_items_like) > 0:
            num_positive_in_top_k_like = len(
                set(top_k_items) & positive_items_like)
            recall_like = num_positive_in_top_k_like / len(positive_items_like)
        else:
            recall_like = 0.0

        # is_follow per user_id
        positive_items_follow = is_follow_dict.get(user_id, set())
        if len(positive_items_follow) > 0:
            num_positive_in_top_k_follow = len(
                set(top_k_items) & positive_items_follow)
            recall_follow = num_positive_in_top_k_follow / \
                len(positive_items_follow)
        else:
            recall_follow = 0.0

        recall_is_like_at_k[user_id] = recall_like
        recall_is_follow_at_k[user_id] = recall_follow

    # avg per user
    avg_recall_like = np.mean(list(recall_is_like_at_k.values()))
    avg_recall_follow = np.mean(list(recall_is_follow_at_k.values()))

    return avg_recall_like, avg_recall_follow


def recall_k_session(unique_user_id, unique_user_embedding, unique_item_id, unique_item_embedding, sample_dict, is_like_dict, is_follow_dict, Ks=[20, 50, 100, 200, 500]):
    """
    Calculate recall@K within inner-session.
    """
    recall_is_like_at_ks = {K: {} for K in Ks}
    recall_is_follow_at_ks = {K: {} for K in Ks}

    item_embedding_matrix = np.array(unique_item_embedding)
    item_ids = np.array(unique_item_id)

    # Inner-session candidate item_id
    user_to_candidate_items = {}
    for user_id, item_id in sample_dict:
        if user_id not in user_to_candidate_items:
            user_to_candidate_items[user_id] = set()
        user_to_candidate_items[user_id].add(item_id)

    for user_idx, user_id in enumerate(unique_user_id):
        # user embedding
        user_embedding = np.array(unique_user_embedding[user_idx])

        candidate_items_id = user_to_candidate_items.get(user_id, set())
        print(
            f"[INFO]Index: {user_idx}, User_id: {user_id}, Inner-Session Candidates Items Cnt: {len(candidate_items_id)}")

        if not candidate_items_id:
            for K in Ks:
                recall_is_like_at_ks[K][user_id] = 0.0
                recall_is_follow_at_ks[K][user_id] = 0.0
            continue

        candidates_item_indices = [np.where(item_ids == item_id)[
            0][0] for item_id in candidate_items_id]
        candidates_item_embedding = item_embedding_matrix[candidates_item_indices]

        # cal similarity
        similarities = np.dot(candidates_item_embedding, user_embedding)

        sorted_indices = np.argsort(-similarities)
        candidate_items_array = np.array(list(candidate_items_id))

        for K in Ks:
            top_k_indices = sorted_indices[:K]
            top_k_items = candidate_items_array[top_k_indices]

            # is_like per user_id
            positive_items_like = is_like_dict.get(user_id, set())
            if len(positive_items_like) > 0:
                num_positive_in_top_k_like = len(
                    set(top_k_items) & positive_items_like)
                recall_like = num_positive_in_top_k_like / \
                    len(positive_items_like)
            else:
                recall_like = 0.0

            # is_follow per user_id
            positive_items_follow = is_follow_dict.get(user_id, set())
            if len(positive_items_follow) > 0:
                num_positive_in_top_k_follow = len(
                    set(top_k_items) & positive_items_follow)
                recall_follow = num_positive_in_top_k_follow / \
                    len(positive_items_follow)
            else:
                recall_follow = 0.0

            recall_is_like_at_ks[K][user_id] = recall_like
            recall_is_follow_at_ks[K][user_id] = recall_follow

    # avg per user
    avg_recall_is_like_at_ks = {K: np.mean(
        list(recall_is_like_at_ks[K].values())) for K in Ks}
    avg_recall_is_follow_at_ks = {K: np.mean(
        list(recall_is_follow_at_ks[K].values())) for K in Ks}

    return avg_recall_is_like_at_ks, avg_recall_is_follow_at_ks


def recall_k_split_embedding(unique_user_id, unique_user_embedding, unique_item_id, unique_item_embedding, sample_dict, is_like_dict, is_follow_dict, Ks=[20, 50, 100, 200, 500]):
    """
    Use the first 32 dimensions and the last 32 dimensions of user_embedding and item_embedding to calculate recall@K for is_like and is_follow, respectively.

    Args:
    - user_embedding_array: An array containing user_id and its corresponding embedding. (shape: [num_users, 65])
    - item_embedding_array: An array containing item_id and its corresponding embedding. (shape: [num_items, 65])
    - sample_label: Arrays containing (user_id, item_id, is_like_label, is_follow_label). (shape: [num_samples, 4])
    - Ks (list): Contains multiple K values that need to be calculated.

    Return:
    - avg_recall_like_at_ks (the first 32 dimensions)
    - avg_recall_like_at_ks (last 32 dimensions)
    """
    user_embeddings_front = unique_user_embedding[:, :32]
    user_embeddings_back = unique_user_embedding[:, 32:]

    item_embeddings_front = unique_item_embedding[:, :32]
    item_embeddings_back = unique_item_embedding[:, 32:]

    # <user_id, item_id> label
    sample_user_ids = sample_dict[:, 0].astype(int)
    sample_item_ids = sample_dict[:, 1].astype(int)

    def compute_recall(user_embeddings, item_embeddings, positive_dict):
        """
        Calculate recall@K.
        """
        recall_at_ks = {K: {} for K in Ks}

        for user_idx, user_id in enumerate(unique_user_id):
            user_embedding = user_embeddings[user_idx]

            candidate_item_indices = np.where(sample_user_ids == user_id)[0]
            candidate_item_ids = sample_item_ids[candidate_item_indices]
            print(
                f"[INFO]Index: {user_idx}, User_id: {user_id}, Inner-Session Candidates Items Cnt: {len(candidate_item_ids)}")

            if len(candidate_item_ids) == 0:
                for K in Ks:
                    recall_at_ks[K][user_id] = 0.0
                continue

            candidate_embeddings = item_embeddings[
                [np.where(unique_item_id == item_id)[0][0]
                 for item_id in candidate_item_ids]
            ]

            similarities = np.dot(candidate_embeddings, user_embedding)

            sorted_indices = np.argsort(-similarities)
            candidate_items_sorted = candidate_item_ids[sorted_indices]

            for K in Ks:
                top_k_items = candidate_items_sorted[:K]
                positive_items = positive_dict.get(user_id, set())
                if len(positive_items) > 0:
                    num_positive_in_top_k = len(
                        set(top_k_items) & positive_items)
                    recall = num_positive_in_top_k / len(positive_items)
                else:
                    recall = 0.0
                recall_at_ks[K][user_id] = recall

        avg_recall_at_ks = {K: np.mean(
            list(recall_at_ks[K].values())) for K in Ks}
        return avg_recall_at_ks

    avg_recall_like_at_ks = compute_recall(
        user_embeddings_front, item_embeddings_front, is_like_dict)
    avg_recall_follow_at_ks = compute_recall(
        user_embeddings_back, item_embeddings_back, is_follow_dict)

    return avg_recall_like_at_ks, avg_recall_follow_at_ks


if __name__ == '__main__':
    recall_user_index_path = sys.argv[1]
    recall_item_index_path = sys.argv[2]
    label_index_path = sys.argv[3]

    user_total_embedding = load_embeddings(recall_user_index_path)
    item_total_embedding = load_embeddings(recall_item_index_path)
    sample_label = load_embeddings(label_index_path)

    # 1. Data Check
    # Embedding Check
    summary_data(item_total_embedding)
    check_consistency_data(item_total_embedding)

    # Lable Check
    summary_label(sample_label)
    check_consistency_label(sample_label)

    # 2. Deduplicate user_embedding / item_embedding
    unique_user_id, unique_user_embedding = deduplicated_data(
        user_total_embedding)
    unique_item_id, unique_item_embedding = deduplicated_data(
        item_total_embedding)
    # Gen sample_id & label
    sample_ids, is_like_dict, is_follow_dict = gen_label_dict(
        sample_label, reverse=True)

    # 3.0 Recall@K
    # avg_recall_like, avg_recall_follow = recall_k(unique_user_id, unique_user_embedding, unique_item_id, unique_item_embedding, is_like_dict, is_follow_dict, K)

    # 3.1. Recall@K Session for Kuairand
    # avg_recall_is_like_at_ks, avg_recall_is_follow_at_ks = recall_k_session(unique_user_id, unique_user_embedding, unique_item_id, unique_item_embedding, sample_ids, is_like_dict, is_follow_dict, [5, 10, 15, 20, 50, 100, 200, 500])

    # 3.1. Recall@K Session for NSB
    avg_recall_is_like_at_ks, avg_recall_is_follow_at_ks = recall_k_split_embedding(
        unique_user_id, unique_user_embedding, unique_item_id, unique_item_embedding, sample_ids, is_like_dict, is_follow_dict, [5, 10, 15, 20, 50, 100, 200, 500])

    print("Average Recall@Ks for is_like:")
    for K, avg_recall in avg_recall_is_like_at_ks.items():
        print(f"Recall@{K}: {avg_recall:.8f}")

    print("\nAverage Recall@Ks for is_follow:")
    for K, avg_recall in avg_recall_is_follow_at_ks.items():
        print(f"Recall@{K}: {avg_recall:.8f}")
