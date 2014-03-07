
### Without familiarization - November 10
betting_no_familiarization = read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/beting_no_fam_25_correct.csv",header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)
boxplot(betting_no_familiarization$Answer.money_allocated_to_target,betting_no_familiarization$Answer.money_allocated_to_logical, betting_no_familiarization$Answer.money_allocated_to_foil)

hist(betting_no_familiarization$Answer.money_allocated_to_target, breaks = 10)
hist(betting_no_familiarization$Answer.money_allocated_to_logical, breaks = 10)
hist(betting_no_familiarization$Answer.money_allocated_to_foil, breaks = 10)

hist(betting_no_familiarization$Answer.money_allocated_to_target - betting_no_familiarization$Answer.money_allocated_to_logical, breaks = 10)
mean(betting_no_familiarization$Answer.money_allocated_to_target - betting_no_familiarization$Answer.money_allocated_to_logical)
sd(betting_no_familiarization$Answer.money_allocated_to_target - betting_no_familiarization$Answer.money_allocated_to_logical)

mean(betting_no_familiarization$Answer.money_allocated_to_target)


### Without familiarization - November 18 - extra "empty familiarization slide"
betting_no_familiarization_Bob_presented = read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/betting_no_fam_november_17_40.csv",header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)
boxplot(betting_no_familiarization_Bob_presented$Answer.money_allocated_to_target, betting_no_familiarization_Bob_presented$Answer.money_allocated_to_logical, betting_no_familiarization_Bob_presented$Answer.money_allocated_to_foil)

plot(jitter(betting_no_familiarization_Bob_presented$Answer.money_allocated_to_target), betting_no_familiarization_Bob_presented$Answer.money_allocated_to_logical)

hist(betting_no_familiarization_Bob_presented$Answer.money_allocated_to_target - betting_no_familiarization_Bob_presented$Answer.money_allocated_to_logical, breaks = 10)
mean(betting_no_familiarization_Bob_presented$Answer.money_allocated_to_target - betting_no_familiarization_Bob_presented$Answer.money_allocated_to_logical)
sd(betting_no_familiarization_Bob_presented$Answer.money_allocated_to_target - betting_no_familiarization_Bob_presented$Answer.money_allocated_to_logical)

mean(betting_no_familiarization_Bob_presented$Answer.money_allocated_to_target)



# December 1
betting_no_familiarization_oneword = read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/betting_no_fam_friend_oneword_1_december_AGHP.csv",header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)
boxplot(betting_no_familiarization_oneword$Answer.money_allocated_to_target, betting_no_familiarization_oneword$Answer.money_allocated_to_logical, betting_no_familiarization_oneword$Answer.money_allocated_to_foil)

hist(betting_no_familiarization_oneword$Answer.money_allocated_to_target, breaks = 30)
hist(betting_no_familiarization_oneword$Answer.money_allocated_to_logical, breaks = 30)
hist(betting_no_familiarization_oneword$Answer.money_allocated_to_foil, breaks = 30)

hist(betting_no_familiarization_oneword$Answer.money_allocated_to_target - betting_no_familiarization_oneword$Answer.money_allocated_to_logical, breaks = 30)
mean(betting_no_familiarization_oneword$Answer.money_allocated_to_target - betting_no_familiarization_oneword$Answer.money_allocated_to_logical)
sd(betting_no_familiarization_oneword$Answer.money_allocated_to_target - betting_no_familiarization_oneword$Answer.money_allocated_to_logical)

mean(betting_no_familiarization_oneword$Answer.money_allocated_to_target)




########### mcf analysis ###########
library(reshape2)
library(ggplot2)

d <- betting_no_familiarization_Bob_presented

###### filter and rearrange
include <- d$Answer.manip_check_target == "\"2\"" & 
  d$Answer.manip_check_dist == "\"1\"" & 
  d$Answer.name_check_correct == "\"TRUE\"" & 
  d$Answer.money_allocated_to_foil < 10

# report back
print(paste(sum(include==FALSE),"subjects excluded, out of",length(include)))

d <- subset(d,include)

####### aggregate and plot
d$target <- d$Answer.money_allocated_to_target
d$logical <- d$Answer.money_allocated_to_logical

md <- melt(d, id.vars="workerid", measure.vars=c("target","logical"),
           variable.name="image", value.name="bet")

ms <- aggregate(bet ~ image, md, mean)
ms$sem <- aggregate(bet ~ image, md, function (x) {sd(x) / sqrt(length(x))})$bet

qplot(bet, fill=image,
      position="dodge",
      geom="histogram",
      data=md)




### With familiarization ~ November 10
betting_yes_familiarization = read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/betting_yes_fam_128_correct.csv",header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)

plot(jitter(betting_yes_familiarization$Answer.familiarization_cond, amount = .08), jitter(betting_yes_familiarization$Answer.money_allocated_to_target, amount = 0.9), col = "red")
points(jitter(betting_yes_familiarization$Answer.familiarization_cond + 0.09, amount = .08), jitter(betting_yes_familiarization$Answer.money_allocated_to_logical, amount = 0.9), col = "blue")
points(jitter(betting_yes_familiarization$Answer.familiarization_cond + 0.18, amount = .08), jitter(betting_yes_familiarization$Answer.money_allocated_to_foil, amount = 0.9), col = "green")

lines(lowess(betting_yes_familiarization$Answer.familiarization_cond, betting_yes_familiarization$Answer.money_allocated_to_target), col="red") # lowess line (x,y)
lines(lowess(betting_yes_familiarization$Answer.familiarization_cond,betting_yes_familiarization$Answer.money_allocated_to_logical), col="blue") # lowess line (x,y)
lines(lowess(betting_yes_familiarization$Answer.familiarization_cond,betting_yes_familiarization$Answer.money_allocated_to_foil), col="green") # lowess line (x,y)

coplot(Answer.money_allocated_to_target ~ Answer.money_allocated_to_logical  | Answer.familiarization_cond, data = betting_yes_familiarization)





