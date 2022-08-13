import torch.nn as nn
# custom loss function for multi-head multi-category classification
def loss_fn(outputs, targets):
    o1, o2, o3, o4, o5, o6, o7 = outputs
    t1, t2, t3, t4, t5, t6, t7 = targets
    l1 = nn.CrossEntropyLoss()(o1, t1)
    l2 = nn.CrossEntropyLoss()(o2, t2)
    l3 = nn.CrossEntropyLoss()(o3, t3)
    l4 = nn.CrossEntropyLoss()(o4, t4)
    l5 = nn.CrossEntropyLoss()(o5, t5)
    l6 = nn.CrossEntropyLoss()(o6, t6)
    l7 = nn.CrossEntropyLoss()(o7, t7)
    return (l1 + l2 + l3 + l4 + l5 + l6 + l7) / 7
    