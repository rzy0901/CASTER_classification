# CASTER_classification

## Source Only + Source mixed target w/ labels

+ Testing accuracies for Schemes 1~6:

<img src="./all_model_test_acc_bar.png" alt="accuracy">

+ Confusion charts for Schemes 1~6:

<table align="center">
  <tr align="center">
    <th>Scheme 1</th>
    <th>Scheme 2</th>
    <th>Scheme 3</th>
  </tr>
  <tr align="center">
    <td><img src="./model1/Output/Test_ConfMatrix.png" alt="scheme 1"></td>
    <td><img src="./model2/Output/Test_ConfMatrix.png" alt="scheme 2"></td>
    <td><img src="./model3/Output/Test_ConfMatrix.png" alt="scheme 3"></td>
  </tr>
  <tr align="center">
    <th>Scheme 4</th>
    <th>Scheme 5</th>
    <th>Scheme 6</th>
  </tr>
  <tr align="center">
    <td><img src="./model4/Output/Test_ConfMatrix.png" alt="scheme 4"></td>
    <td><img src="./model5/Output/Test_ConfMatrix.png" alt="scheme 5"></td>
    <td><img src="./model6/Output/Test_ConfMatrix.png" alt="scheme 6"></td>
  </tr>
</table>

## Unsupervised domain adpataion

<table>
<thead>
  <tr align="center">
    <th colspan="2">Source only (<a href="./features.ipynb" alt="tsne">features.ipynb</a>: 83%)</th>
    <th colspan="2">Transferred (<a href="./ADDA_test_new.ipynb" alt="tsne">ADDA_test_new.ipynb</a>: 93%)</th>
  </tr>
</thead>
<tbody>
  <tr align="center">
    <td><img src="./model1/Output/Test_ConfMatrix.png" alt="scheme 1"></td>
    <td><img src="./model1/Output/TSNE.png" alt="TSNE"></td>
    <td><img src="./output_DA2/Test_ConfMatrix.png" alt=""></td>
    <td><img src="./output_DA2/ALL_TSNE.png" alt=""><</td>
  </tr>
</tbody>
</table>