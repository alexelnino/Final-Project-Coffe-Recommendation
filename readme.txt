Memprediksi kopi yang sesuai dengan properties yang user inginkan dengan metode random forest.
dengan 90 % train dan 10 % tes

Alasan:(perbandingan antara logistic regression, decision tree, random forest, k-nearest neighbour, support vector machine)
1. Logistic Regression memiliki score terendah
2. KNN cenderung memiliki score tertinggi dan cenderung sama dengan SVM
3. decision tree dan random forest memiliki score ditengah-tengah tetapi decision tree cenderung memprediksi hasil yang berbeda dari metode yang lain

membuat grafik yang membandingkan properties antara arabica dan robusta
=>tipe grafik : radar chart
=>menggunakan pyplot.xticks

Tampilan:
1. HomePage
![Image1](https://ibb.co/9NgdBY0)
2. Chart properties arabica vs robusta
![Image2](https://ibb.co/Xkx02fG)
3. prediction page
![Image3](https://ibb.co/crgByLt)
5. Result after input prediction
![Image4](https://ibb.co/YcnvzZH)
Arabika
row: 1308
44 kolom data mentah

Robusta
row:29

7 fitur x
1 fitur y

#['Arabica' 'Robusta']
#[    0        1     ]     


Logistic Regression :  96.2687 %
[1]
[0]
Decision Tree :  100.0 %
[0]
[0]
RF :  100.0 %
[0]
[0]
SVM :  98.5075 %
[1]
[0]
KNN :  99.2537 %
[1]
[0]

sumber : https://github.com/jldbc/coffee-quality-database
These data were collected from the Coffee Quality Institute's review pages in January 2018.
https://database.coffeeinstitute.org