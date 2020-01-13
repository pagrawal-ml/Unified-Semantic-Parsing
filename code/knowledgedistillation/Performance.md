Noting the performance for the different datasets

## knowledge distillation performance (single domain) 

Dataset | Teacher Match Accuracy  | Teacher Accuracy with denotation | Teacher Network Information | Teacher Epochs | Student Match Accuracy  | Student Accuracy with denotation | Student Network Information | Student Epochs | 
---|---|---|---|---|---|---|---|---
recipes | 0.7361  |  | default network | 90 | 67.1% | Hard: 0.7176 Soft: 0.7692 | default network | 90 
recipes | 0.7685  | Hard: 0.8241 Soft: 0.8632 | default network | 200 |  0.704 | Hard: 0.7454 Soft: 0.7809 | default network | 90
recipes | 0.7685  | Hard: 0.8241 Soft: 0.8632 | default network | 200 |  0.620 | Hard : 0.6713 Soft: 0.7114 | rnn_size=100 | 90 
recipes | 0.7453  | Hard: 0.8241 Soft: 0.8681 | layer_2 | 200 |  0.699 | Hard: 0.7638 Soft: 0.8035|  default network| 90
recipes | 0.7453  | Hard: 0.8241 Soft: 0.8681 | layer_2 | 200 |  0.718 | Hard: 0.7778 Soft: 0.8238|  default network| 200
recipes | 0.7453  | Hard: 0.8241 Soft: 0.8681 | layer_2 | 200 |  0.611 | Hard: 0.6574 Soft: 0.7099 | rnn_size=100 | 90
recipes | 0.7037  | Hard: 0.7685 Soft: 0.8151 | layer_3 | 200 |  0.667| Hard: 0.7315 Soft: 0.7732 |  default network| 90
recipes | 0.7037  | Hard: 0.7685 Soft: 0.8151 | layer_3 | 200 |  0.644 | Hard: 0.6944 Soft: 0.7399 | rnn_size=100 | 90
recipes | 0.7361  | Hard: 0.7917 Soft: 0.8392 | layer_4 | 200 |  0.593| Hard: 0.6481 Soft: 0.6847 |  default network| 90
recipes | 0.7361  | Hard: 0.7917 Soft: 0.8392 | layer_4 | 200 |  0.685 | Hard: 0.7314 Soft: 0.7751 | rnn_size=100 | 90
recipes | 0.7361  | Hard: 0.8102 Soft: 0.8529 | layer_4,size=400 | 200 |  |  | default network | 90
recipes | 0.7361  | Hard: 0.8102 Soft: 0.8529 | layer_4,size=400 | 200 |  |  | rnn_size=100 | 90

## knowledge distillation performance (domain agnostic multi-domain) 

Domains | Dataset | Teacher Match Accuracy  | Teacher Accuracy with denotation | Teacher Network Information | Teacher Epochs | Student Match Accuracy  | Student Accuracy with denotation | Student Network Information | Student Epochs | 
---|---|---|---|---|---|---|---|---|---
Recipe, Restaurant | Recipe | 0.7314 | Hard: 0.7824 Soft: 0.8190 | default_network | 200 |.653 |Hard: 0.7129, Soft: 0.7651 | default_network | 200 |
Recipe, Restaurant | Restaurant | 0.5241 | Hard: 0.7560 Soft: 0.8037 | default_network | 200 | .476 | Hard: 0.7108 Soft: 0.7480 | default_network | 200 |
Recipe, Restaurant | Recipe | 0.6342 | Hard: 0.7222 Soft: 0.7666  | layer_2 | 200 | | | default_network | 200 |
Recipe, Restaurant | Restaurant |  0.5271| Hard: 0.7801 Soft: 0.8273 | layer_2 | 200 | | | default_network | 200 |
Recipe_Pruned, Restaurant_pruned | Recipe |0.7638 | Hard: 0.8148 Soft: 0.8556 | default_network | 200 | | | default_network | 200 |
Recipe_Pruned, Restaurant_pruned | Restaurant | 0.5271 | Hard: 0.8133 Soft: 0.8541| default_network | 200 | | | default_network | 200 |
Recipe_Pruned, Restaurant_pruned  | Recipe | 0.7593 | Hard: 0.8194 Soft: 0.8538 | layer_2 | 200 | Avg: 0.75 | Avg: Hard 0.815, Soft 0.862 | default_network | 200 |
Recipe_Pruned, Restaurant_pruned  | Restaurant |  0.5361| Hard: 0.8012 Soft: 0.8405| layer_2 | 200 | Avg: 0.553 | Avg: Hard 0.807, Soft 0.845 | default_network | 200 |

## knowledge distillation performance (domain agnostic multi-domain) 
Domains | Dataset | Teacher Match Accuracy  | Teacher Accuracy with denotation | Teacher Network Information | Teacher Epochs | Student Match Accuracy  | Student Accuracy with denotation | Student Epochs | Student Network Information | 
---|---|---|---|---|---|---|---|---|---
all | socialnetwork | 0.6844 | Hard: 0.8032 Soft: 0.8813|  layer_2 | 200 | 22.4% | Hard: 0.2647 Soft: 0.2833 |200 | default_network 
all | socialnetwork |  |  | | | 70.9% |Hard: 0.8179 Soft: 0.8888 |600 | layer_2,rnn_size=300|
all | socialnetwork |  |  | | | 69% |Hard: 0.8066 Soft: 0.8746 |600 | layer_2,rnn_size=300 (repeat)|
all | recipes |  0.7315 | Hard: 0.8009 Soft: 0.8422 | layer_2 | 200 | 73.1% | Hard: 0.7777 Soft: 0.8228 | 200| default_network
all | recipes |  |  | | |74.5%| Hard: 0.8102 Soft: 0.8524 |600 | layer_2,rnn_size=300|
all | recipes |  |  | | |75.5%| Hard: 0.8241 Soft: 0.8620 |600 | layer_2,rnn_size=300 (repeat)|
all | restaurants |  0.5361| Hard: 0.7982 Soft: 0.8334 | layer_2 | 200 | 51%| Hard: 0.7289 Soft: 0.7804 | 200| default_network
all | restaurants |  |  | | | 55.7% | Hard: 0.8133 Soft: 0.8487 | 600 | layer_2,rnn_size=300|
all | restaurants |  |  | | | 55.4% | Hard: 0.8012 Soft: 0.8331 |600 | layer_2,rnn_size=300 (repeat)|
all | publications |  0.6024 | Hard: 0.7453 Soft: 0.7849 | layer_2 | 200 | 67.7% | Hard: 0.7701 Soft: 0.8213 | 200 |default_network
all | publications |  |  | | |  66.5% | Hard: 0.7619 Soft: 0.8308 |600 | layer_2,rnn_size=300 |
all | publications |  |  | | |  65.8% | Hard: 0.7702 Soft: 0.8185 |600 | layer_2,rnn_size=300 (repeat)|
all | calendar |  0.5893 | Hard: 0.8036 Soft: 0.8758  | layer_2 | 200 | 45% |Hard: 0.6786 Soft: 0.7573 | 200| default_network
all | calendar |  |  | | | 62.5% | Hard: 0.8095 Soft: 0.8815 |600 | layer_2,rnn_size=300|
all | calendar |  |  | | | 57.1% | Hard: 0.8155 Soft: 0.8826 |600 | layer_2,rnn_size=300 (repeat)|
all | blocks |  0.3809| Hard: 0.5589 Soft: 0.6483 |layer_2 | 200 | 38% | Hard: 0.5238 Soft: 0.5901 | 200 |default_network
all | blocks |  |  | | |39.1% | Hard: 0.5439 Soft: 0.6201 |600 | layer_2,rnn_size=300|
all | blocks |  |  | | |42.1% | Hard: 0.5564 Soft: 0.6294| 600 | layer_2,rnn_size=300 (repeat)|
all | basketball |  0.8030 | Hard: 0.8312 Soft: 0.8946 | layer_2 | 200 | 43% | Hard: 0.4322 Soft: 0.4425 |200 | default_network
all | basketball |  |  | | | 82.4% | Hard: 0.8439 Soft: 0.8870 |600 | layer_2,rnn_size=300|
all | basketball |  |  | | | 80.3% | Hard: 0.8261 Soft: 0.8819 |600 | layer_2,rnn_size=300 (repeat)|
all | housing |  45.50% | Hard: 0.7037 Soft: 0.7748| layer_2 | 200 | 67.7% | Hard: 0.6402 Soft: 0.7301 |200 |default_network
all | housing |  |  | | |52.4% | Hard: 0.7619 Soft: 0.8308 |600 | layer_2,rnn_size=300 |
all | housing |  |  | | |52.9% | Hard: 0.7249 Soft: 0.7976 |600 | layer_2,rnn_size=300(repeat)|
average |---| 0.597825 | Hard: 0.7556 Soft:0.8169 |---|---|50.99 | Hard: 0.602025   Soft: 0.65347|---|---


## knowledge distillation performance (domain agnostic multi-domain -  5 domains) 
 Domains | Dataset | Teacher Match Accuracy  | Teacher Accuracy with denotation | Teacher Network Information | Teacher Epochs | Student Match Accuracy  | Student Accuracy with denotation | Student Network Information | Student Epochs | 
 ---|---|---|---|---|---|---|---|---|---
+all | socialnetwork | 0.6844 | Hard: 0.8032 Soft: 0.8813|  NA | NA | NA|NA |NA |
+all | recipes |  0.7315 | Hard: 0.8009 Soft: 0.8422 | layer_2 | 200 | 73.6% | Hard: 0.8009 Soft: 0.8537| 200|
+all | restaurants |  0.5361| Hard: 0.7982 Soft: 0.8334 | layer_2 | 200 | 53.9%| Hard: 0.8042 Soft: 0.8461| 200|
+all | publications |  0.6024 | Hard: 0.7453 Soft: 0.7849 | layer_2 | 200 | 67.1% |Hard: 0.7950 Soft: 0.8194 | 200|
+all | calendar |  0.5893 | Hard: 0.8036 Soft: 0.8758  | layer_2 | 200 | 61.9% |Hard: 0.8155 Soft: 0.8952 | 200|
+all | blocks |  0.3809| Hard: 0.5589 Soft: 0.6483 | NA | NA | NA|NA | NA|
+all | basketball |  0.8030 | Hard: 0.8312 Soft: 0.8946 | NA | NA | NA |NA |NA |
+all | housing |  0.4550 | Hard: 0.7037 Soft: 0.7748| layer_2 | 200 | 52.4% | Hard: 0.7407 Soft: 0.8089|200 |
