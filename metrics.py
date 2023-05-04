from scipy.stats import mode

def main_metric(tracking_ids):
    tr_ids = 0
    len_all_ids = 0

    for true_id in tracking_ids:
        track_ids = tracking_ids[true_id]
        track_ids = [i if i != None else -1 for i in track_ids]
        tr_id = mode(track_ids).mode[0]
        if tr_id == -1:
            tr_id_len = 0
        else:
            tr_id_len = mode(track_ids).count[0]

        tr_ids += tr_id_len
        len_all_ids += len(track_ids)

    return tr_ids / len_all_ids