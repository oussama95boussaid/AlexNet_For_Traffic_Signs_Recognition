# AlexNet

AlexNet is a popular base network for transfer learning because its structure is relatively straightforward,
it's not too big, and it performs well empirically.

AlexNet puts the network on two GPUs, which allows for building a larger network. Although most of the calculations are done in parallel, the GPUs communicate with each other in certain layers. The original research paper on AlexNet said that parallelizing the network decreased the classification error rate by 1.7% when compared to a neural network that used half as many neurons on one GPU.

<img src="/repository/assets/employee.png" alt="Employee data" title="Employee Data title">
