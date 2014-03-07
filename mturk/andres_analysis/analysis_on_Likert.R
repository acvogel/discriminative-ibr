

### Likert with Bob being presented

Likert_no_familiarization_Bob_presented = read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/likert_no_fam_november_18.csv",header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)


boxplot(Likert_no_familiarization_Bob_presented$Answer.Likert_value_target, Likert_no_familiarization_Bob_presented$Answer.Likert_value_logical, Likert_no_familiarization_Bob_presented$Answer.Likert_value_foil)
hist(Likert_no_familiarization_Bob_presented$Answer.Likert_value_target)
hist(Likert_no_familiarization_Bob_presented$Answer.Likert_value_logical)
hist(Likert_no_familiarization_Bob_presented$Answer.Likert_value_foil)

hist(Likert_no_familiarization_Bob_presented$Answer.Likert_value_target - Likert_no_familiarization_Bob_presented$Answer.Likert_value_logical)
mean(Likert_no_familiarization_Bob_presented$Answer.Likert_value_target - Likert_no_familiarization_Bob_presented$Answer.Likert_value_logical)
table(Likert_no_familiarization_Bob_presented$Answer.Likert_value_target - Likert_no_familiarization_Bob_presented$Answer.Likert_value_logical)
cor(Likert_no_familiarization_Bob_presented[,c(33, 45, 41)])


# Taking out the non-1 foils
nonOneLikert = Likert_no_familiarization_Bob_presented[Likert_no_familiarization_Bob_presented$Answer.Likert_value_foil == 1,]
hist(nonOneLikert$Answer.Likert_value_target - nonOneLikert$Answer.Likert_value_logical)
table(nonOneLikert$Answer.Likert_value_target - nonOneLikert$Answer.Likert_value_logical)
mean(nonOneLikert$Answer.Likert_value_target - nonOneLikert$Answer.Likert_value_logical)
sd(nonOneLikert$Answer.Likert_value_target - nonOneLikert$Answer.Likert_value_logical)







# 29 Nov 2013
Likert_no_familiarization_Bob_presented = read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/likert_no_fam_november_29.csv",header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)





library(scatterplot3d)
attach(mtcars)
scatterplot3d(jitter(Likert_no_familiarization_Bob_presented$Answer.Likert_value_target),jitter(Likert_no_familiarization_Bob_presented$Answer.Likert_value_logical),jitter(Likert_no_familiarization_Bob_presented$Answer.Likert_value_foil), highlight.3d=TRUE, main="3D Scatterplot", type="h")

plot(jitter(Likert_no_familiarization_Bob_presented$Answer.Likert_value_target), jitter(Likert_no_familiarization_Bob_presented$Answer.Likert_value_logical))
lines(lowess(Likert_no_familiarization_Bob_presented$Answer.Likert_value_target, Likert_no_familiarization_Bob_presented$Answer.Likert_value_logical), col="red") # lowess line (x,y)

coplot(Answer.Likert_value_target ~ Answer.Likert_value_logical  | Answer.Likert_value_foil, data = Likert_no_familiarization_Bob_presented)


summary(lm(Answer.Likert_value_target ~ Answer.Likert_value_foil * Answer.Likert_value_logical, data = Likert_no_familiarization_Bob_presented))