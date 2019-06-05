
data = scan() #read data 
time_series_data = ts(data)
time_seties_data

# 3 groups of 12 
factor = rep(1:3, each = 12)
oneway.test(data ~ factor, var.equal =T)

diff_data = diff(data)
diff_factor = factor[1:35]
oneway.test(diff_data ~ diff_factor, var.equal =T)

model = arima(data, order = c(2,1,10), seasonal = c(1,0,2))
r = resid(model)
ft = fitted(model)

ts.plot(data)
lines(ft,col="red")
data.frame(data,ft)

accuracy(model)

#R suggestions ... 

forecast(auto.arima(diff_data))
plot(forecast(auto.arima(diff_data)))

