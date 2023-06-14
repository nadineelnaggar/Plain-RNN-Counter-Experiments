import pandas as pd
import re

data = "Accuracy for epoch 0=0.38%, avg train loss = 0.33410219561308624 num_correct = 38, train val loss = 9.486708045005798e-06, train val acc = 0.38%, val loss = 2.014426290988922e-05, val accuracy = 0.1%, long val loss = 6.304391622543334e-05, long val acc = 0.0%, time = 1m 29.05s
Accuracy for epoch 1=0.38%, avg train loss = 0.321389909170568 num_correct = 38, train val loss = 1.4013651013374329e-05, train val acc = 0.38%, val loss = 1.1409323662519455e-05, val accuracy = 0.1%, long val loss = 6.0320138931274415e-05, long val acc = 0.0%, time = 2m 59.53s 
Accuracy for epoch 2=0.38%, avg train loss = 0.3213909192353487 num_correct = 38, train val loss = 1.126997321844101e-05, train val acc = 0.38%, val loss = 2.5205719470977783e-05, val accuracy = 0.1%, long val loss = 6.155911684036255e-05, long val acc = 0.0%, time = 4m 30.28s 
Accuracy for epoch 3=0.38%, avg train loss = 0.3213531729370356 num_correct = 38, train val loss = 9.157851338386535e-06, train val acc = 0.38%, val loss = 2.477654963731766e-05, val accuracy = 0.1%, long val loss = 6.462032198905945e-05, long val acc = 0.0%, time = 6m 0.9s 
Accuracy for epoch 4=0.38%, avg train loss = 0.3213370733201504 num_correct = 38, train val loss = 1.036849394440651e-05, train val acc = 0.38%, val loss = 2.33852818608284e-05, val accuracy = 0.1%, long val loss = 7.617202401161194e-05, long val acc = 0.0%, time = 7m 31.58s 
Accuracy for epoch 5=0.38%, avg train loss = 0.32137599227130415 num_correct = 38, train val loss = 9.370235353708266e-06, train val acc = 0.38%, val loss = 1.9986237585544586e-05, val accuracy = 0.1%, long val loss = 6.170960664749145e-05, long val acc = 0.0%, time = 9m 2.9s 
Accuracy for epoch 6=0.38%, avg train loss = 0.32140361025929454 num_correct = 38, train val loss = 1.0703517496585845e-05, train val acc = 0.38%, val loss = 2.6615262031555174e-05, val accuracy = 0.1%, long val loss = 6.115158200263976e-05, long val acc = 0.0%, time = 10m 33.98s 
Accuracy for epoch 7=0.38%, avg train loss = 0.3213565817639232 num_correct = 38, train val loss = 9.23914909362793e-06, train val acc = 0.38%, val loss = 2.0157746970653534e-05, val accuracy = 0.1%, long val loss = 7.398122549057007e-05, long val acc = 0.0%, time = 12m 5.46s 
Accuracy for epoch 8=0.38%, avg train loss = 0.3213592697292566 num_correct = 38, train val loss = 8.745357394218444e-06, train val acc = 0.38%, val loss = 2.3674522340297698e-05, val accuracy = 0.1%, long val loss = 5.748317241668701e-05, long val acc = 0.0%, time = 13m 36.36s 
Accuracy for epoch 9=0.38%, avg train loss = 0.3213868456959724 num_correct = 38, train val loss = 1.2273260205984115e-05, train val acc = 0.38%, val loss = 2.8775253891944884e-05, val accuracy = 0.1%, long val loss = 6.443520188331604e-05, long val acc = 0.0%, time = 15m 7.6s 
Accuracy for epoch 10=0.38%, avg train loss = 0.321359876832366 num_correct = 38, train val loss = 9.74656566977501e-06, train val acc = 0.38%, val loss = 2.6512914896011353e-05, val accuracy = 0.1%, long val loss = 5.4583245515823365e-05, long val acc = 0.0%, time = 16m 38.52s 
Accuracy for epoch 11=0.38%, avg train loss = 0.321394705401361 num_correct = 38, train val loss = 9.184084832668304e-06, train val acc = 0.38%, val loss = 2.5879499316215515e-05, val accuracy = 0.1%, long val loss = 6.148203015327454e-05, long val acc = 0.0%, time = 18m 9.56s 
Accuracy for epoch 12=0.38%, avg train loss = 0.3214020552441478 num_correct = 38, train val loss = 1.0565485060214996e-05, train val acc = 0.38%, val loss = 1.7083889245986937e-05, val accuracy = 0.1%, long val loss = 6.546743512153626e-05, long val acc = 0.0%, time = 19m 40.98s 
Accuracy for epoch 13=0.38%, avg train loss = 0.3213634154945612 num_correct = 38, train val loss = 1.224740818142891e-05, train val acc = 0.38%, val loss = 2.048906236886978e-05, val accuracy = 0.1%, long val loss = 6.28623366355896e-05, long val acc = 0.0%, time = 21m 12.09s 
Accuracy for epoch 14=0.38%, avg train loss = 0.32136845554113386 num_correct = 38, train val loss = 1.0445938259363174e-05, train val acc = 0.38%, val loss = 3.278077244758606e-05, val accuracy = 0.1%, long val loss = 7.373327612876892e-05, long val acc = 0.0%, time = 22m 42.96s 
Accuracy for epoch 15=0.38%, avg train loss = 0.3213193482413888 num_correct = 38, train val loss = 1.426689475774765e-05, train val acc = 0.38%, val loss = 1.9715738296508788e-05, val accuracy = 0.1%, long val loss = 5.882473587989807e-05, long val acc = 0.0%, time = 24m 13.88s 
Accuracy for epoch 16=0.38%, avg train loss = 0.32137874187529086 num_correct = 38, train val loss = 1.129673719406128e-05, train val acc = 0.38%, val loss = 2.0982842147350312e-05, val accuracy = 0.1%, long val loss = 6.711181402206421e-05, long val acc = 0.0%, time = 25m 44.92s 
Accuracy for epoch 17=0.38%, avg train loss = 0.32136945270597933 num_correct = 38, train val loss = 1.3908948004245757e-05, train val acc = 0.38%, val loss = 1.5827476978302002e-05, val accuracy = 0.1%, long val loss = 6.968857645988464e-05, long val acc = 0.0%, time = 27m 15.81s 
Accuracy for epoch 18=0.38%, avg train loss = 0.32135899882763624 num_correct = 38, train val loss = 1.2052717059850693e-05, train val acc = 0.38%, val loss = 2.0615965127944945e-05, val accuracy = 0.1%, long val loss = 6.656743288040161e-05, long val acc = 0.0%, time = 28m 47.52s 
Accuracy for epoch 19=0.38%, avg train loss = 0.32140206291228535 num_correct = 38, train val loss = 1.1904255300760269e-05, train val acc = 0.38%, val loss = 2.338142395019531e-05, val accuracy = 0.1%, long val loss = 5.7095086574554445e-05, long val acc = 0.0%, time = 30m 19.01s 
Accuracy for epoch 20=0.38%, avg train loss = 0.3213696453511715 num_correct = 38, train val loss = 1.008613333106041e-05, train val acc = 0.38%, val loss = 2.508270740509033e-05, val accuracy = 0.1%, long val loss = 5.9580260515213015e-05, long val acc = 0.0%, time = 31m 50.46s 
Accuracy for epoch 21=0.38%, avg train loss = 0.32137097230404615 num_correct = 38, train val loss = 1.4598466455936432e-05, train val acc = 0.38%, val loss = 1.630604714155197e-05, val accuracy = 0.1%, long val loss = 6.922949552536011e-05, long val acc = 0.0%, time = 33m 21.52s 
Accuracy for epoch 22=0.38%, avg train loss = 0.3213716320067644 num_correct = 38, train val loss = 1.2818531692028046e-05, train val acc = 0.38%, val loss = 2.7731603384017944e-05, val accuracy = 0.1%, long val loss = 7.859677672386169e-05, long val acc = 0.0%, time = 34m 52.87s 
Accuracy for epoch 23=0.38%, avg train loss = 0.3213984665587544 num_correct = 38, train val loss = 1.2010338902473449e-05, train val acc = 0.38%, val loss = 2.7729555964469908e-05, val accuracy = 0.1%, long val loss = 6.110886335372925e-05, long val acc = 0.0%, time = 36m 24.35s 
Accuracy for epoch 24=0.38%, avg train loss = 0.32141201327592134 num_correct = 38, train val loss = 1.445426493883133e-05, train val acc = 0.38%, val loss = 2.365884631872177e-05, val accuracy = 0.1%, long val loss = 6.53603434562683e-05, long val acc = 0.0%, time = 37m 55.8s 
Accuracy for epoch 25=0.38%, avg train loss = 0.32137291306853294 num_correct = 38, train val loss = 8.42454433441162e-06, train val acc = 0.38%, val loss = 2.191890776157379e-05, val accuracy = 0.1%, long val loss = 6.306757330894471e-05, long val acc = 0.0%, time = 39m 27.04s 
Accuracy for epoch 26=0.38%, avg train loss = 0.3213732033133507 num_correct = 38, train val loss = 9.462503343820571e-06, train val acc = 0.38%, val loss = 2.1521741151809693e-05, val accuracy = 0.1%, long val loss = 6.816752552986145e-05, long val acc = 0.0%, time = 40m 59.34s 
Accuracy for epoch 27=0.38%, avg train loss = 0.3214043212428689 num_correct = 38, train val loss = 1.396503895521164e-05, train val acc = 0.38%, val loss = 1.703714281320572e-05, val accuracy = 0.1%, long val loss = 6.794481873512268e-05, long val acc = 0.0%, time = 42m 31.67s 
Accuracy for epoch 28=0.38%, avg train loss = 0.3213347306370735 num_correct = 38, train val loss = 1.2532229721546173e-05, train val acc = 0.38%, val loss = 3.376443684101105e-05, val accuracy = 0.1%, long val loss = 6.649368405342102e-05, long val acc = 0.0%, time = 44m 3.18s 
Accuracy for epoch 29=0.38%, avg train loss = 0.32137310009747744 num_correct = 38, train val loss = 8.938683569431305e-06, train val acc = 0.38%, val loss = 1.5907394886016846e-05, val accuracy = 0.1%, long val loss = 5.575768351554871e-05, long val acc = 0.0%, time = 49m 22.23s "

for line in data:
    split_string = line.split(', time = ')[0]

print(data)


pattern = re.compile(r"Accuracy for epoch (\d+)=([\d.]+)%.*, avg train loss = ([\d.]), train val loss = ([\d.e-]+), train val acc = ([\d.]+)%, val loss = ([\d.e-]+), val acc = ([\d.]+)%, long val loss = ([\d.e-]+), long val acc = ([\d.]+)%")

matches = re.findall(pattern, data)
print(matches)
df = pd.DataFrame(matches, columns=["Epoch", "Avg Train Loss", "Accuracy", "Train Val Loss", "Train Val Accuracy", "Val Loss", "Val Accuracy", "Long Val Loss", "Long Val Accuracy"])

print(df)
# df.to_excel("data.xlsx", index=False)