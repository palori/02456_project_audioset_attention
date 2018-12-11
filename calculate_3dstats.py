def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 3d array, (samples_num, time, classes_num)
      target: 3d array, (samples_num, time, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    timestep_num = target.shape[-2]
    stats = []

    # Class-wise statistics
    for j,k for k in range(classes_num) for j in range(timestep_num):

        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, j, k], output[:, j, k], average=None)

        # AUC
        auc = metrics.roc_auc_score(target[:, j, k], output[:, j, k], average=None)

        # Precisions, recalls
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            target[:, j, k], output[:, j, k])

        # FPR, TPR
        (fpr, tpr, thresholds) = metrics.roc_curve(target[:, j, k], output[:, j, k])

        save_every_steps = 1000     # Sample statistics to reduce size
        dict = {'precisions': precisions[0::save_every_steps],
                'recalls': recalls[0::save_every_steps],
                'AP': avg_precision,
                'fpr': fpr[0::save_every_steps],
                'fnr': 1. - tpr[0::save_every_steps],
                'auc': auc}
        stats.append(dict)

    return stats