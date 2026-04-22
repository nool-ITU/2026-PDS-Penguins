# upload file
df <- read.csv(file.choose())
names(df)
table(df$Class)


df$Class <- as.factor(df$Class)
# create a model using feature
model <- glm(Class ~  Compactness+ Asymmetry+ Convexity+ Multicolor+Hue_Var+ Sat_Var+ Val_Var+ Hue_Entropy,
             data = df, 
             family = binomial)

# Guarda il risultato (quello che cercavi!)
summary(model)
