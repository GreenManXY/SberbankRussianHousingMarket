setwd("F:/My shit/Forskning/Kaggle/Russian Housing")

# Wall material: 1 - panel, 2 - brick, 3 - wood, 4 - mass concrete, 5 - breezeblock, 
# 6 - mass concrete plus brick.

# ID_railroad_terminal: These should be: Moscow Kazanskaya railway station, Moscow Kiyevskaya railway 
# station, 
# Moscow Kurskaya railway station, Moscow Paveletskaya railway station, Moscow Rizhskaya railway station, 
# Moscow Savyolovskaya railway station, Moscow Smolenskaya railway station, Moscow Yaroslavskaya 
# railway station.

# kitch_sq not included in life_sq

# install.packages("fscaret", dependencies = c("Depends", "Suggests"))

library(ggplot2)
library(readr)
library(dplyr)
library(MASS)
library(caret)
library(fscaret)
library(randomForest)
library(xgboost)
library(e1071)
library(doSNOW)
library(lubridate)

# Useful functions
show.n <- function(x) {
  return (c(y = median(x)*1.05, label = length(x)))
}

show.median <- function(x) {
  return (c(y = 0, label = median(x)))
}

nzmean <- function(x) {
  zvals <- x == 0
  if (all(zvals)) 0 else mean(x[!zvals], na.rm = TRUE)
}

rmsle <- function(x,y) {sqrt(mean((log1p(x)-log1p(y))^2))}
rmse <- function(x,y) {sqrt(mean((x-y)^2))}

# Read and combine data
train <- read.csv("F:/My shit/Forskning/Kaggle/Russian Housing/train.csv", stringsAsFactors = FALSE)
test <- read.csv("F:/My shit/Forskning/Kaggle/Russian Housing/test.csv", stringsAsFactors = FALSE)
# fix <- read.csv2("F:/My shit/Forskning/Kaggle/Russian Housing/fix.csv", sep = ";", row.names = NULL,
#                  stringsAsFactors = FALSE)
macro <- read.csv("F:/My shit/Forskning/Kaggle/Russian Housing/macro.csv", stringsAsFactors = FALSE)

test$price_doc <- rep(NA, 7662)
houseDF <- rbind(train, test)

# Replace erronious values with fix
# houseDF[fix$id, names(fix)] <- fix
# rm(fix)

# Check for duplicates
# nrow(train) - nrow(unique(train))
# nrow(test) - nrow(unique(test))

# NA count
# r <- as.character(seq(1,length(names(houseDF)),1))
# na_count <- as.data.frame(sapply(houseDF, function(y) sum(is.na(y))), col.names = c("NAs"))
# na_count <- cbind(missing = na_count, r)
# rm(r)

# Create building_id based on location
i <- which(names(houseDF) %in% c("mkad_km", "ttk_km", "sadovoe_km", "bulvar_ring_km", "kremlin_km"))
houseDF$sum_km <- rowSums(houseDF[, i])
houseDF <- transform(houseDF, building_id = as.numeric(interaction(sub_area, sum_km, drop = TRUE)))
rm(i)

# full_sq (use 10m2 as cutoff for invalid data for full_sq and 5m2 as a cutoff for life_sq)
houseDF$full_sq[which(houseDF$full_sq < 10)] <- NA
i <- which(is.na(houseDF$full_sq) & !is.na(houseDF$life_sq) & houseDF$life_sq > 5)
houseDF$full_sq[i] <- houseDF$life_sq[i]
rm(i)

# Several entries in the dataset seem to omit the decimal so that, for example, 
# there is an full_sq = 5326.
# The cutoff value for unreasonable full_sq is here assumed to be 325m2, provided life_sq != NA.
# When life_sq isn't NA, move the decimal for full_sq by dividing with 100 or 10.
houseDF$life_sq[which(houseDF$life_sq <= 5)] <- NA # convert invalid values to NA
i_1 <- which(houseDF$full_sq > 1000 & !is.na(houseDF$life_sq))
houseDF$full_sq[i_1] <- houseDF$full_sq[i_1] / 100
i_2 <- which(houseDF$full_sq > 325 & !is.na(houseDF$life_sq))
houseDF$full_sq[i_2] <- houseDF$full_sq[i_2] / 10
rm(i_1, i_2)

houseDF <- houseDF %>%
  group_by(building_id) %>%
  mutate(full_sq = ifelse(is.na(full_sq), median(full_sq, na.rm = TRUE), full_sq))

houseDF <- houseDF %>%
  group_by(building_id) %>%
  mutate(full_sq = ifelse(full_sq > 325, median(full_sq, na.rm = TRUE), full_sq))
houseDF <- houseDF[order(houseDF$id),]

# life_sq
# Many life_sq seem to be exactly 10 times full_sq, which seems to be a decimal error
houseDF <- houseDF %>%
  mutate(life_sq = ifelse(life_sq == 10*full_sq , life_sq/10, life_sq))

# Some life_sq are larger than full_sq but seem to have invalid values, which I asumme to be 
# at 100m2 cutoff
i_1 <- which(houseDF$life_sq > 1000 & houseDF$life_sq > houseDF$full_sq)
houseDF$life_sq[i_1] <- houseDF$life_sq[i_1] / 100
i_2 <- which(houseDF$life_sq > 100 & houseDF$life_sq > houseDF$full_sq)
houseDF$life_sq[i_2] <- houseDF$life_sq[i_2] / 10
rm(i_1, i_2)

# Many life_sq < 100m2 seem to be exchanged with full_sq, so I reverse that
temp <- which(houseDF$life_sq > houseDF$full_sq)
temp2 <- houseDF$life_sq[temp]
houseDF$life_sq[temp] <- houseDF$full_sq[temp]
houseDF$full_sq[temp] <- temp2
rm(temp, temp2)

# ggplot(houseDF, aes(x = full_sq, y = life_sq)) +
#   geom_point() +
#   geom_smooth(method = "lm")

i <- which(is.na(houseDF$life_sq))
life_sq.model <- rlm(life_sq ~ full_sq, data = houseDF)
life_sq.pred <- predict(life_sq.model, houseDF)
houseDF$life_sq[i] <- life_sq.pred[i]
rm(life_sq.model, life_sq.pred, i)

# max_floor, assuming that 99 and above are errors
houseDF$max_floor <- as.numeric(houseDF$max_floor)
houseDF$floor <- as.numeric(houseDF$floor)
houseDF$max_floor[which(houseDF$max_floor == 0)] <- NA
houseDF$max_floor[which(houseDF$max_floor >= 99)] <- NA

houseDF <- houseDF %>%
  group_by(building_id) %>%
  mutate(max_floor = ifelse(is.na(max_floor), max(max_floor, na.rm = TRUE), max_floor)) %>%
  mutate(max_floor = ifelse(is.na(max_floor), max(floor, na.rm = TRUE), max_floor))
houseDF <- houseDF[order(houseDF$id),]
houseDF$max_floor[which(houseDF$max_floor == 0)] <- NA

houseDF <- houseDF %>%
  group_by(sub_area) %>%
  mutate(max_floor = ifelse(is.na(max_floor), median(max_floor, na.rm = TRUE), max_floor))
houseDF <- houseDF[order(houseDF$id),]

# max_floor above 1 and smaller than floor seems exchanged, I switch them back
temp <- which(houseDF$max_floor < houseDF$floor & houseDF$max_floor > 1)
temp2 <- houseDF$floor[temp]
houseDF$floor[temp] <- houseDF$max_floor[temp]
houseDF$max_floor[temp] <- temp2
rm(temp, temp2)

# max_floor = 1 and smaller than floor seems like an error, set equal to max floor in building
i <- which(houseDF$max_floor < houseDF$floor & houseDF$max_floor == 1)
houseDF[i,] <- houseDF[i,] %>%
  group_by(building_id) %>%
  mutate(max_floor = max(floor, na.rm = TRUE))
houseDF <- houseDF[order(houseDF$id),]
rm(i)

# ggplot(houseDF, aes(x = as.factor(max_floor), y = price_doc)) +
#   geom_boxplot() +
#   stat_summary(fun.data = show.n, geom = "text", fun.y = median) +
#   stat_summary(fun.data = show.median, geom = "text")

# floor
houseDF$floor[which(houseDF$floor == 0)] <- NA # I'm assuming that floor = 0 is actually NA
houseDF <- houseDF %>%
  mutate(floor = ifelse(is.na(floor), ceiling(max_floor/2), floor)) # Set missing floor to 
# half of max_floor
houseDF <- houseDF[order(houseDF$id),]

# ggplot(houseDF, aes(x = max_floor , y = floor)) +
#   geom_point() +
#   geom_smooth(method = "lm")

# material
houseDF$material <- as.numeric(houseDF$material)
houseDF <- houseDF %>%
  group_by(building_id) %>%
  mutate(material = ifelse(is.na(material), floor(median(material, na.rm = TRUE)), material))

houseDF <- houseDF %>%
  group_by(sub_area) %>%
  mutate(material = ifelse(is.na(material), ceiling(median(material, na.rm = TRUE)), material))
houseDF <- houseDF[order(houseDF$id),]

# build_year
houseDF$build_year <- as.numeric(houseDF$build_year)
houseDF$build_year[which(houseDF$build_year <= 1)] <- NA

# Obvious manual mistakes, can be fixed by formula but then gives lower score on Leaderboard
houseDF$build_year[which(houseDF$build_year == 20052009)] <- 2009
houseDF$build_year[which(houseDF$build_year == 4965)] <- 1965
houseDF$build_year[which(houseDF$build_year == 1691)] <- 1991
houseDF$build_year[which(houseDF$build_year == 215)] <- 2015
houseDF$build_year[which(houseDF$build_year == 71)] <- 1971
houseDF$build_year[which(houseDF$build_year < 1860 | houseDF$build_year > 2020)] <- NA

houseDF <- houseDF %>%
  group_by(building_id) %>%
  mutate(build_year = ifelse(is.na(build_year), median(build_year, na.rm = TRUE), build_year))

houseDF <- houseDF %>%
  group_by(sub_area) %>%
  mutate(build_year = ifelse(is.na(build_year), median(build_year, na.rm = TRUE), build_year))
houseDF <- houseDF[order(houseDF$id),]

houseDF$build_year <- floor(houseDF$build_year)

# num_room
houseDF$num_room <- as.numeric(houseDF$num_room)
houseDF$num_room[which(houseDF$num_room == 0)] <- NA

# Create a feature, room_sq and correct units with more than 10 rooms but room_sq < 5
# Also, num_room 1 and life_sq > 50 is probably an error
houseDF$room_sq <- houseDF$life_sq / houseDF$num_room
houseDF$num_room[which(houseDF$num_room >= 10 & houseDF$room_sq < 5)] <- NA
houseDF$num_room[which(houseDF$num_room == 1 & houseDF$life_sq > 50)] <- NA

# ggplot(houseDF, aes(x = as.factor(num_room), y = life_sq)) +
#   geom_boxplot() +
#   stat_summary(fun.data = show.n, geom = "text", fun.y = median) +
#   stat_summary(fun.data = show.median, geom = "text") +
#   scale_y_continuous(limits = c(0,150), breaks = seq(0,150,10))

r <- boxplot(life_sq ~ as.factor(num_room), data = houseDF)
houseDF <- houseDF %>%
  mutate(num_room = ifelse(is.na(num_room) & life_sq < r$stats[2,2], 1, num_room)) %>%
  mutate(num_room = ifelse(is.na(num_room) & life_sq < r$stats[2,3], 2, num_room)) %>%
  mutate(num_room = ifelse(is.na(num_room) & life_sq < r$stats[2,4], 3, num_room)) %>%
  mutate(num_room = ifelse(is.na(num_room) & life_sq <= r$stats[4,4], 4, num_room)) %>%
  mutate(num_room = ifelse(is.na(num_room) & life_sq > r$stats[4,4], 5, num_room))

rm(r)

# kitch_sq
houseDF$kitch_sq <- as.numeric(houseDF$kitch_sq)
houseDF$kitch_sq[which(houseDF$kitch_sq <= 1)] <- NA
houseDF$kitch_sq[which(houseDF$kitch_sq >= houseDF$full_sq)] <- NA
houseDF$kitch_sq[which(houseDF$kitch_sq >= houseDF$life_sq)] <- NA

houseDF$rel_kitch_sq <- houseDF$kitch_sq / houseDF$full_sq

# ggplot(houseDF, aes(x = full_sq, y = rel_kitch_sq)) +
#   geom_point() +
#   geom_smooth(method = "nls", se = FALSE, 
#               method.args = list(formula = y ~ a/x + b, start = list(a = 1, b = 0.1)))

nls <- nls(rel_kitch_sq ~ a/full_sq + b, data = houseDF, start = list(a = 1, b = 0.1))
nls.pred <- predict(nls, houseDF)

i <- which(is.na(houseDF$rel_kitch_sq))
houseDF$rel_kitch_sq[i] <- nls.pred[i]
houseDF$kitch_sq[i] <- houseDF$rel_kitch_sq[i] * houseDF$full_sq[i]
rm(i, nls, nls.pred)

# state
houseDF$state <- as.numeric(houseDF$state)
houseDF$state[which(houseDF$state == 33)] <- 3 # Correct manual error

houseDF <- houseDF %>%
  group_by(building_id) %>%
  mutate(state = ifelse(is.na(state), floor(median(state, na.rm = TRUE)), state))

houseDF <- houseDF %>%
  group_by(sub_area) %>%
  mutate(state = ifelse(is.na(state), ceiling(median(state, na.rm = TRUE)), state))

houseDF <- houseDF %>%
  group_by(build_year) %>%
  mutate(state = ifelse(is.na(state), floor(median(state, na.rm = TRUE)), state))

# ggplot(houseDF, aes(x = as.factor(state), y = build_year)) +
#   geom_boxplot() +
#   stat_summary(fun.data = show.n, geom = "text", fun.y = median) +
#   stat_summary(fun.data = show.median, geom = "text") +
#   scale_y_continuous(limits = c(1800,2020), breaks = seq(1800,2020,20))

# product_type
houseDF$product_type[which(houseDF$product_type == "Investment")] <- 2
houseDF$product_type[which(houseDF$product_type == "OwnerOccupier")] <- 1
houseDF$product_type <- as.numeric(houseDF$product_type)

houseDF <- houseDF %>%
  group_by(building_id) %>%
  mutate(product_type = ifelse(is.na(product_type), median(product_type, na.rm = TRUE), product_type))

houseDF <- houseDF %>%
  group_by(sub_area) %>%
  mutate(product_type = ifelse(is.na(product_type), median(product_type, na.rm = TRUE), product_type))
houseDF <- houseDF[order(houseDF$id),]

# sub_area
sub_area_price <- houseDF[1:30471,] %>%
  group_by(sub_area) %>%
  summarise(m2_median = median(price_doc/full_sq)) %>%
  arrange(m2_median)

## Encode districts as numbers based on avg price
sub_area_price <- sub_area_price[order(sub_area_price$m2_median),]
for (i in 1:length(sub_area_price$sub_area)) {
  houseDF$sub_area[which(houseDF$sub_area == sub_area_price$sub_area[i])] <- i
}
houseDF$sub_area <- as.integer(houseDF$sub_area)
rm(i, sub_area_price)

# preschool_education_centers_raion
houseDF$preschool_education_centers_raion <- ifelse(houseDF$preschool_education_centers_raion == 1 &
                                                      houseDF$preschool_quota == 0, 
                                                    0,
                                                    houseDF$preschool_education_centers_raion)

# preschool_quota
houseDF$preschool_quota <- ifelse(houseDF$preschool_education_centers_raion == 0 &
                                    is.na(houseDF$preschool_quota), 
                                  0, 
                                  houseDF$preschool_quota)

# school_quota
houseDF$school_quota <- ifelse(houseDF$school_education_centers_raion == 0 &
                                 is.na(houseDF$school_quota), 
                               0, 
                               houseDF$school_quota)

# hospital_beds_raion, I'm assuming that NA = 0
houseDF$hospital_beds_raion[which(is.na(houseDF$hospital_beds_raion))] <- 0

# convert yes/no to 1/0 depedning on which variable has higher median price_doc
yes.no_2.1 <- c("culture_objects_top_25", "thermal_power_plant_raion", "radiation_raion", 
                "railroad_terminal_raion", "nuclear_reactor_raion", "detention_facility_raion", 
                "big_road1_1line") 

no.yes_2.1 <- c("incineration_raion", "oil_chemistry_raion", "big_market_raion", "water_1line", 
                "railroad_1line")

for (i in yes.no_2.1) {
  houseDF[which(houseDF[, i] == "yes"), i] <- 1
  houseDF[which(houseDF[, i] == "no"), i] <- 0
}
houseDF[, yes.no_2.1] <- lapply(houseDF[, yes.no_2.1], FUN = as.numeric)

for (i in no.yes_2.1) {
  houseDF[which(houseDF[, i] == "yes"), i] <- 0
  houseDF[which(houseDF[, i] == "no"), i] <- 1
}
houseDF[, no.yes_2.1] <- lapply(houseDF[, no.yes_2.1], FUN = as.numeric)

rm(i, yes.no_2.1, no.yes_2.1)

# check population sums = looks good
# houseDF$full_all_diff <- houseDF$full_all - (houseDF$male_f + houseDF$female_f)
# houseDF$age_group_diff <- houseDF$full_all - (houseDF$young_all + houseDF$work_all + 
#                                               houseDF$ekder_all)

# raion_build_count_with_material_info
r <- c("raion_build_count_with_material_info", "build_count_block", 
       "build_count_wood", "build_count_frame", "build_count_brick", 
       "build_count_monolith", "build_count_panel", "build_count_foam", 
       "build_count_slag", "build_count_mix", "raion_build_count_with_builddate_info", 
       "build_count_before_1920", "build_count_1921.1945", "build_count_1946.1970", 
       "build_count_1971.1995", "build_count_after_1995")

# for(i in r) {
#   print(
#     ggplot(houseDF, aes(x = houseDF[, i], y = price_doc)) +
#       geom_point() +
#       ggtitle(i)
#   )
# }

houseDF[, r] <- lapply(houseDF[, r], FUN = as.numeric)
houseDF[, r] <- lapply(houseDF[, r], FUN = function(x) ifelse(is.na(x), 0, x))
rm(r)

# metro_min_walk and metro_min_km, replace with average
walk.car.km.ratio <- mean(houseDF$metro_km_walk / houseDF$metro_km_avto, na.rm = TRUE)
walk.speed.avg <- mean(houseDF$metro_min_walk / houseDF$metro_km_walk, na.rm = TRUE)

i <- which(is.na(houseDF$metro_km_walk & !is.na(houseDF$metro_km_avto)))
houseDF$metro_km_walk[i] <- houseDF$metro_km_avto[i] * walk.car.km.ratio

i <- which(is.na(houseDF$metro_min_walk & !is.na(houseDF$metro_km_walk)))
houseDF$metro_min_walk[i] <- houseDF$metro_km_walk[i] * walk.speed.avg

# railroad_station_walk_km and railroad_station_walk_min and ID-railroad_station_walk
walk.railroad.km.ratio <- mean(houseDF$railroad_station_walk_km / houseDF$railroad_station_avto_km, 
                               na.rm = TRUE)
walk.speed.avg2 <- mean(houseDF$railroad_station_walk_min / houseDF$railroad_station_walk_km, 
                        na.rm = TRUE)

i <- which(is.na(houseDF$railroad_station_walk_km & !is.na(houseDF$railroad_station_avto_km)))
houseDF$railroad_station_walk_km[i] <- houseDF$railroad_station_avto_km[i] * walk.railroad.km.ratio

i <- which(is.na(houseDF$railroad_station_walk_min & !is.na(houseDF$railroad_station_walk_km)))
houseDF$railroad_station_walk_min[i] <- houseDF$railroad_station_walk_km[i] * walk.speed.avg2

i <- which(is.na(houseDF$ID_railroad_station_walk) & !is.na(houseDF$ID_railroad_station_avto))
houseDF$ID_railroad_station_walk[i] <- houseDF$ID_railroad_station_avto[i]

rm(walk.car.km.ratio, walk.railroad.km.ratio, walk.speed.avg, walk.speed.avg2, i)

# ecology
houseDF$ecology <- lapply(houseDF$ecology, switch, 'excellent' = 4, 'good' = 3, 'satisfactory' = 2, 
                          'poor' = 1, 'no data' = 0)
houseDF$ecology <- as.integer(houseDF$ecology)

# cafe_count_500: replace NAs with -1 where there is no cafe or replace with mean when there is
cafe_500 <- c("cafe_sum_500_min_price_avg", "cafe_sum_500_max_price_avg", "cafe_avg_price_500")
houseDF[which(houseDF$cafe_count_500 == 0), cafe_500] <- 0

i <- which(houseDF$cafe_count_500 > 0)
houseDF[i,] <- houseDF[i,] %>%
  group_by(sub_area) %>%
  mutate_at(cafe_500, funs(if_else(is.na(.), as.integer(mean(., na.rm = TRUE)), as.integer(.))))

rm(cafe_500, i)

# cafe_count_1000: replace NAs with -1 where there is no cafe or replace with mean when there is
cafe_1000 <- c("cafe_sum_1000_min_price_avg", "cafe_sum_1000_max_price_avg", "cafe_avg_price_1000")
houseDF[which(houseDF$cafe_count_1000 == 0), cafe_1000] <- 0

i <- which(houseDF$cafe_count_1000 > 0)
houseDF[i, ] <- houseDF[i,] %>%
  group_by(sub_area) %>%
  mutate_at(cafe_1000, funs(if_else(is.na(.), as.integer(mean(., na.rm = TRUE)), as.integer(.))))

rm(cafe_1000, i)

# cafe_count_1500: replace NAs with -1 where there is no cafe or replace with mean when there is
cafe_1500 <- c("cafe_sum_1500_min_price_avg", "cafe_sum_1500_max_price_avg", "cafe_avg_price_1500")
houseDF[which(houseDF$cafe_count_1500 == 0), cafe_1500] <- 0

i <- which(houseDF$cafe_count_1500 > 0)
houseDF[i, ] <- houseDF[i,] %>%
  group_by(sub_area) %>%
  mutate_at(cafe_1500, funs(if_else(is.na(.), as.integer(mean(., na.rm = TRUE)), as.integer(.))))

rm(cafe_1500, i)

# cafe_count_2000: replace NAs with -1 where there is no cafe or replace with mean when there is
cafe_2000 <- c("cafe_sum_2000_min_price_avg", "cafe_sum_2000_max_price_avg", "cafe_avg_price_2000")
houseDF[which(houseDF$cafe_count_2000 == 0), cafe_2000] <- 0

i <- which(houseDF$cafe_count_2000 > 0)
houseDF[i, ] <- houseDF[i,] %>%
  group_by(sub_area) %>%
  mutate_at(cafe_2000, funs(if_else(is.na(.), as.integer(mean(., na.rm = TRUE)), as.integer(.))))

rm(cafe_2000, i)

# cafe_count_3000: replace NAs with -1 where there is no cafe or replace with mean when there is
cafe_3000 <- c("cafe_sum_3000_min_price_avg", "cafe_sum_3000_max_price_avg", "cafe_avg_price_3000")
houseDF[which(houseDF$cafe_count_3000 == 0), cafe_3000] <- 0

i <- which(houseDF$cafe_count_3000 > 0)
houseDF[i, ] <- houseDF[i,] %>%
  group_by(sub_area) %>%
  mutate_at(cafe_3000, funs(if_else(is.na(.), as.integer(mean(., na.rm = TRUE)), as.integer(.))))

rm(cafe_3000, i)

# cafe_count_5000: replace NAs with -1 where there is no cafe or replace with mean when there is
cafe_5000 <- c("cafe_sum_5000_min_price_avg", "cafe_sum_5000_max_price_avg", "cafe_avg_price_5000")
houseDF[which(houseDF$cafe_count_5000 == 0), cafe_5000] <- 0

i <- which(houseDF$cafe_count_5000 > 0)
houseDF[i, ] <- houseDF[i,] %>%
  group_by(sub_area) %>%
  mutate_at(cafe_5000, funs(if_else(is.na(.), as.integer(mean(., na.rm = TRUE)), as.integer(.))))
houseDF <- houseDF[order(houseDF$id),]
rm(cafe_5000, i)

# Use known cafe avg prices to replace remaining NAs
min_avg_price <- c("cafe_sum_500_min_price_avg", "cafe_sum_1000_min_price_avg", 
                   "cafe_sum_1500_min_price_avg", "cafe_sum_2000_min_price_avg", 
                   "cafe_sum_3000_min_price_avg", "cafe_sum_5000_min_price_avg")
max_avg_price <- c("cafe_sum_500_max_price_avg", "cafe_sum_1000_max_price_avg", 
                   "cafe_sum_1500_max_price_avg", "cafe_sum_2000_max_price_avg", 
                   "cafe_sum_3000_max_price_avg", "cafe_sum_5000_max_price_avg")
avg_price <- c("cafe_avg_price_500", "cafe_avg_price_1000", "cafe_avg_price_1500", 
               "cafe_avg_price_2000", "cafe_avg_price_3000", "cafe_avg_price_5000")
b <- c(min_avg_price, max_avg_price, avg_price)

w <- houseDF[,b]
w$min_mean <- apply(w[,min_avg_price], 1, nzmean)
w$max_mean <- apply(w[,max_avg_price], 1, nzmean)
w$avg_mean <- apply(w[,avg_price], 1, nzmean)

w <- w %>%
  rowwise() %>%
  mutate_at(min_avg_price, funs(ifelse(is.na(.), min_mean, .))) %>%
  mutate_at(max_avg_price, funs(ifelse(is.na(.), max_mean, .))) %>%
  mutate_at(avg_price, funs(ifelse(is.na(.), avg_mean, .)))

houseDF[, b] <- w[, b]

rm(b, min_avg_price, max_avg_price, avg_price, w)

temp <- mean(houseDF$green_part_2000 / houseDF$green_part_3000, trim = 0.05, na.rm = TRUE)
i <- which(is.na(houseDF$green_part_2000))
houseDF$green_part_2000[i] <- houseDF$green_part_3000[i] * temp
rm(temp, i)

# prom_part_5000
temp <- c("prom_part_500", "prom_part_1000", "prom_part_1500", "prom_part_2000", "prom_part_3000")
i <- which(is.na(houseDF$prom_part_5000 & rowSums(houseDF[,which(names(houseDF) %in% temp)]) == 0))
houseDF$prom_part_5000[i] <- 0
rm(temp, i)

# Convert timestamp to numeric
houseDF$timestamp <- as.Date(houseDF$timestamp, "%Y-%m-%d")
houseDF$num_days <- difftime(houseDF$timestamp, houseDF$timestamp[1], units = "days")
houseDF$num_days <- as.numeric(houseDF$num_days)

# Extract useful features from macro data
## Fix out of bounds date and commas
macro2 <- macro[which(macro$timestamp < "2016-05-31"),]
macro2 <- as.data.frame(sapply(macro2, function(x) gsub(",", ".", x)), stringsAsFactors = FALSE)
timestampX <- macro2[, 1]
macro2 <- macro2[, c(2:100)]
macro2 <- as.data.frame(sapply(macro2, as.numeric, stringsAsFactors = FALSE))
macro2 <- cbind(timestamp = timestampX, macro2)

## Create number of days column to match houseDF
macro2$timestamp <- as.Date(macro2$timestamp, "%Y-%m-%d")
macro2$num_days <- difftime(macro2$timestamp, houseDF$timestamp[1], units = "days")
macro2$num_days <- as.numeric(macro2$num_days)

## Number of days with offsets
macro2$num_days_month_offset <- difftime(macro2$timestamp, houseDF$timestamp[1] %m-% months(1),
                                         units = "days")
macro2$num_days_month_offset <- as.numeric(macro2$num_days_month_offset)

macro2$num_days_quarter_offset <- difftime(macro2$timestamp, houseDF$timestamp[1] %m-% months(3),
                                         units = "days")
macro2$num_days_quarter_offset <- as.numeric(macro2$num_days_quarter_offset)

macro2$num_days_6month_offset <- difftime(macro2$timestamp, houseDF$timestamp[1] %m-% months(6),
                                         units = "days")
macro2$num_days_6month_offset <- as.numeric(macro2$num_days_6month_offset)

macro2$num_days_annual_offset <- difftime(macro2$timestamp, houseDF$timestamp[1] %m-% months(12),
                                         units = "days")
macro2$num_days_annual_offset <- as.numeric(macro2$num_days_annual_offset)

## Extrapolate with approximations
macro2 <- macro2 %>%
  mutate_all(funs(ifelse(is.na(.), predict(loess(. ~ num_days, data = macro2, 
                                                 control = loess.control(surface = "direct")), 
                                           macro2), .)))

monthly <- c("cpi", "ppi", "balance_trade", "deposits_growth", "net_capital_export", "deposits_rate",
             "mortgage_growth", "income_per_cap")
quarterly <- c("gdp_quart", "gdp_quart_growth", "balance_trade_growth", 
               "average_provision_of_build_contract_moscow")
anually <- c("gdp_deflator", "grp", "grp_growth", "real_dispos_income_per_cap_growth", "salary",
             "salary_growth", "retail_trade_turnover", "retail_trade_turnover_per_cap",
             "retail_trade_turnover_growth", "labor_force", "unemployment", "employment",
             "invest_fixed_capital_per_cap", "invest_fixed_assets", "profitable_enterpr_share",
             "unprofitable_enterpr_share", "share_own_revenues", "overdue_wages_per_cap",
             "fin_res_per_cap", "marriages_per_1000_cap", "divorce_rate", "construction_value",
             "invest_fixed_assets_phys", "pop_natural_increase", "pop_migration", "pop_total_inc",
             "childbirth", "mortality", "housing_fund_sqm", 
             "lodging_sqm_per_cap", "water_pipes_share", "baths_share", "sewerage_share", 
             "gas_share", "hot_water_share", "electric_stove_share", "heating_share", 
             "old_house_share", "average_life_exp", "infant_mortarity_per_1000_cap", 
             "perinatal_mort_per_1000_cap", "incidence_population", "rent_price_4.room_bus", 
             "rent_price_3room_bus", "rent_price_2room_bus", "rent_price_1room_bus", 
             "rent_price_3room_eco", "rent_price_2room_eco", "rent_price_1room_eco", 
             "load_of_teachers_preschool_per_teacher", "child_on_acc_pre_school", 
             "load_of_teachers_school_per_teacher", "students_state_oneshift", 
             "modern_education_share", "old_education_build_share", "provision_doctors", 
             "provision_nurse", "load_on_doctors", "power_clinics", "hospital_beds_available_per_cap", 
             "hospital_bed_occupancy_per_year", "provision_retail_space_sqm", 
             "provision_retail_space_modern_sqm", "turnover_catering_per_cap", 
             "theaters_viewers_per_1000_cap", "seats_theather_rfmin_per_100000_cap", 
             "museum_visitis_per_100_cap", "bandwidth_sports", "population_reg_sports_share", 
             "students_reg_sports_share", "apartment_build", "apartment_fund_sqm")

macro2$timestamp <- as.Date(timestampX)
macro2 <- macro2 %>%
  mutate(month = format(timestamp, "%m"), year = format(timestamp, "%Y")) %>%
  group_by(year, month) %>%
  mutate_at(monthly, mean)

macro2 <- macro2 %>%
  group_by(year) %>%
  mutate_at(anually, mean)

macro2 <- macro2 %>%
  mutate(quarter = quarter(timestamp)) %>%
  group_by(year, quarter) %>%
  mutate_at(quarterly, mean)

rm(monthly, quarterly, anually, timestampX)

## Join macro2 data with houseDF
a <- houseDF
houseDF <- inner_join(houseDF, macro2, by = "num_days")

# Join macro2 with monthly, quarterly and 6 months lag
DF_offsetM <- a
names(DF_offsetM)[297] <- "num_days_month_offset"
DF_offsetM <- inner_join(DF_offsetM, macro2, by = "num_days_month_offset")

DF_offsetQ <- a
names(DF_offsetQ)[297] <- "num_days_quarter_offset"
DF_offsetQ <- inner_join(DF_offsetQ, macro2, by = "num_days_quarter_offset")

DF_offset6M <- a
names(DF_offset6M)[297] <- "num_days_6month_offset"
DF_offset6M <- inner_join(DF_offset6M, macro2, by = "num_days_6month_offset")

DF_offsetY <- a
names(DF_offsetY)[297] <- "num_days_annual_offset"
DF_offsetY <- inner_join(DF_offsetY, macro2, by = "num_days_annual_offset")

# Are there any offset features from macro2 that improve the model?
feat <- read.csv("F:/My shit/Forskning/Kaggle/Russian Housing/LM features.csv",
                 stringsAsFactors = FALSE)
feat <- feat$x

feat_m <- read.csv("F:/My shit/Forskning/Kaggle/Russian Housing/LM features monthly offset.csv",
                   stringsAsFactors = FALSE)
feat_m <- feat_m$x
feat_m <- feat_m[which(feat_m %in% names(macro2))]
feat_m <- feat_m[-which(feat_m %in% c("num_days"))]

feat_q <- read.csv("F:/My shit/Forskning/Kaggle/Russian Housing/LM features quarterly offset.csv",
                   stringsAsFactors = FALSE)
feat_q <- feat_q$x
feat_q <- feat_q[which(feat_q %in% names(macro2))]

feat_6m <- read.csv("F:/My shit/Forskning/Kaggle/Russian Housing/LM features 6M offset.csv",
                    stringsAsFactors = FALSE)
feat_6m <- feat_6m$x
feat_6m <- feat_6m[which(feat_6m %in% names(macro2))]
feat_6m <- feat_6m[-which(feat_6m %in% c("num_days"))]

feat_y <- read.csv("F:/My shit/Forskning/Kaggle/Russian Housing/LM features annual offset.csv",
                   stringsAsFactors = FALSE)
feat_y <- feat_y$x
feat_y <- feat_y[which(feat_y %in% names(macro2))]

# Join offset values to houseDF
houseDF <- bind_cols(houseDF, DF_offsetM[, feat_m])
names(houseDF)[405] <- "net_capital_export_monthly_offset"

houseDF <- bind_cols(houseDF, DF_offsetQ[, feat_q])
names(houseDF)[406:412] <- c("oil_urals_quarterly_offset", "gdp_annual_quarterly_offset", 
                             "deposits_growth_quarterly_offset", "deposits_rate_quarterly_offset",
                             "mortgage_value_quarterly_offset", "grp_quarterly_offset", 
                             "income_per_cap_quarterly_offset")

houseDF <- bind_cols(houseDF, DF_offset6M[, feat_6m])
names(houseDF)[413:416] <- c("gdp_quart_6m_offset", "ppi_6m_offset", "eurrub_6m_offset",
                             "average_provision_of_build_contract_moscow_6m_offset")

houseDF <- bind_cols(houseDF, DF_offsetY[, feat_y])
names(houseDF)[417:421] <- c("gdp_quart_growth_annual_offset", "eurrub_annual_offset",
                             "gdp_annual_annual_offset", "deposits_growth_annual_offset",
                             "mortgage_growth_annual_offset")

rm(DF_offsetM, DF_offsetQ, DF_offset6M, DF_offsetY, feat, feat_m, feat_q, feat_6m, feat_y)

# Create a m2 price. Use building average when there are 10 or more transactions per building.
a2 <- houseDF
houseDF$m2price_area <- NA
houseDF$m2price_train <- houseDF$price_doc / houseDF$full_sq

houseDF <- houseDF %>%
  group_by(building_id) %>%
  mutate(m2price_area = ifelse(n() >= 10, median(m2price_train, na.rm = TRUE), m2price_area))

houseDF <- houseDF %>%
  group_by(sub_area) %>%
  mutate(m2price_area = ifelse(is.na(m2price_area), median(m2price_train, na.rm = TRUE),
                               m2price_area))

houseDF <- houseDF[, -which(names(houseDF) %in% c("m2price_train"))]

# Other new features
# houseDF$top_floor <- ifelse(houseDF$floor == houseDF$max_floor, 1, 0)
houseDF$room_sq <- houseDF$life_sq / houseDF$num_room
# 
# i <- houseDF$big_church_count_1000 + houseDF$big_church_count_2000
# houseDF$has_big_church <- ifelse(i != 0, 1, 0)
# rm(i)
# 
# houseDF$has_sport_object <- ifelse(houseDF$sport_objects_raion != 0, 1, 0)
# houseDF$has_shopping_centers <- ifelse(houseDF$shopping_centers_raion != 0, 1, 0)
# houseDF$big_house <- ifelse(houseDF$full_sq >= 70, 1, 0)
# houseDF$small_house <- ifelse(houseDF$full_sq < 70, 1, 0)
# houseDF$tiny_kitch <- ifelse(houseDF$kitch_sq <= 10, 1, 0)

# houseDF$has_culture_obj_raion <- ifelse(houseDF$culture_objects_top_25_raion != 0, 1, 0)
# houseDF$has_top_university <- ifelse(houseDF$university_top_20_raion != 0, 1, 0)
# houseDF$product_state <- houseDF$product_type * houseDF$state
# 
# i <- houseDF$cafe_count_1000 + houseDF$cafe_count_2000 + houseDF$cafe_count_3000
# houseDF$has_cafe <- ifelse(i != 0, 1, 0)
# rm(i)

# houseDF$has_big_market <- ifelse(houseDF$big_market_raion != 0, 1, 0)
# 
# i <- houseDF$church_count_500 + houseDF$church_count_2000 + houseDF$church_count_5000
# houseDF$has_church <- ifelse(i != 0, 1, 0)
# rm(i)

# Select important features with LM
# features <- names(houseDF)
# features <- features[-which(features %in% c("price_doc", "id", "timestamp.x", "timestamp.y",
#                                             "year", "month", "quarter", "num_days_6month_offset",
#                                             "num_days_annual_offset", "num_days_quarter_offset",
#                                             "num_days_month_offset"))]
# 
# train2 <- houseDF[1:30471, c(features, "price_doc")]
# train2$price_doc <- log1p(train2$price_doc)
# 
# partition <- createDataPartition(y = train2$price_doc, p = 0.75, list = FALSE)
# training <- train2[partition, ]
# testing <- train2[-partition, ]
# 
# label.train <- training[, "price_doc"]

# RF model
# rf.feat <- rfcv(training, label.train, cv.fold = 10, scale = "log", step = 0.5,
#               mtry = function(p) max(1, floor(sqrt(p))), recursive = TRUE)
# with(rf.feat, plot(n.var, error.cv, log="x", type="o", lwd=2))

# rf <- randomForest(price_doc ~ ., data = training, ntree = 500)
# rf.imp.feat <- as.data.frame(rf$importance)
# rf.imp.feat$variables <- rownames(rf.imp.feat)
# rf.imp.feat <- cbind(rf.imp.feat[order(rf.imp.feat$IncNodePurity, decreasing = TRUE),], 
#                      n = seq(1,415,1))
# 
# write.csv(rf.imp.feat$variables[1:208], "RF features.csv")

# Check colinearity
# features <- read.csv("F:/My shit/Forskning/Kaggle/Russian Housing/RF features.csv",
#                      stringsAsFactors = FALSE)
# features <- features$x
# 
# library(usdm)
# DF <- as.data.frame(houseDF[1:30471, features], stringsAsFactor = FALSE)
# v <- vif(DF)
# v1 <- vifcor(DF, th = 0.7)
# v2 <- vifstep(DF, th = 10)
# 
# v_feat <- c(as.character(v1@results$Variables), as.character(v2@results$Variables))
# v_feat <- unique(v_feat)
# write.csv(v_feat, "VIF features.csv", row.names = FALSE)

# Visualise features
# houseDFx <- houseDF2[, c(features2, "price_doc")]
# 
# for (i in features2) {
#   print (
#   ggplot(houseDF2, aes(x = as.factor(state), y = price_doc)) +
    # geom_boxplot() +
    # stat_summary(fun.data = show.n, geom = "text", fun.y = median) +
    # stat_summary(fun.data = show.median, geom = "text") +
    # theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)) +
#     ggtitle(names(houseDFx)[i])
#   )
# }
# 
# ggplot(houseDF2, aes(x = life_sq, y = kitch_sq)) +
#   geom_point() +
#   scale_x_continuous(limits = c(0,60), breaks = seq(0,340,20)) +
#   geom_smooth(method = "loess") +
#   ggtitle("kitch_sq")

# Check and deal with outliers
train.temp <- houseDF[c(1:30471),]
test.temp <- houseDF[c(30472:38133),]

i <- which(train.temp$price_doc %in% c(990000, 1000000, 2000000, 3000000))
train.temp <- train.temp[-i,]
train.temp <- train.temp[which(train.temp$price_doc < 100000000),]

houseDF2 <- as.data.frame(rbind(train.temp, test.temp), stringsAsFactors = FALSE)
rm(i, test.temp)

# Modelling
features <- read.csv("F:/My shit/Forskning/Kaggle/Russian Housing/RF features.csv",
                      stringsAsFactors = FALSE)
features2 <- features$x

# Check correlations
# corr.data <- as.data.frame(houseDF[1:length(train.temp$price_doc), c(features2, "price_doc")])
# corr <- cor(corr.data)
# write.csv(corr, "Useful features.csv")

# houseDFx <- train2[, c(features2, "price_doc")]
# 
# for (i in 150:174) {
#   print (
#     ggplot(houseDFx, aes(x = as.factor(houseDFx[, i]), y = price_doc)) +
#       geom_boxplot() +
#       stat_summary(fun.data = show.n, geom = "text", fun.y = median) +
#       stat_summary(fun.data = show.median, geom = "text") +
#       theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)) +
#       ggtitle(names(houseDFx[i]))
#   )
# }

train2 <- houseDF2[1:length(train.temp$price_doc), c(features2, "price_doc")]

train2p <- train2[-which(names(train2) %in% c("id", "price_doc"))]
preproc <- preProcess(train2p, method = c("BoxCox", "center", "scale"))
train2p <- predict(preproc, train2p)

train2p$price_doc <- log1p(train2$price_doc)

set.seed(1234)
partition <- createDataPartition(y = train2p$price_doc, p = 0.75, list = FALSE)
training <- train2p[partition, ]
testing <- train2p[-partition, ]

label.train <- training[, "price_doc"]
label.test <- testing[, "price_doc"]

# LM model
lm <- lm(price_doc ~ ., data = training)
lm.pred <- predict(lm, testing)
rmse(lm.pred, label.test)

lm.output <- as.data.frame(cbind(price_doc = label.test, PredPrice = lm.pred))
ggplot(lm.output) +
  geom_point(aes(x = price_doc, y = price_doc)) +
  geom_point(aes(x = price_doc, y = PredPrice), colour = "red") +
  ggtitle("LM model") + ylab("price_doc/PredPrice")

# XGBoost Model
training2 <- as.matrix(training[, features2], rownames.force = NA)
testing2 <- as.matrix(testing[, features2], rownames.force = NA)

training2 <- as(training2, "sparseMatrix")
testing2 <- as(testing2, "sparseMatrix")

xgb.train <- xgb.DMatrix(data = training2, label = label.train)
xgb.test <- xgb.DMatrix(data = testing2, label = label.test)

xgb_params = list(seed = 1234, booster = "gbtree", objective = "reg:linear", eta = 0.3, gamma = 10,
                  max_depth = 6, min_child_weight = 1, subsample = 1, colsample_bytree = 1)

bst <- xgb.cv(params = xgb_params, data = xgb.train, nrounds = 40, nfold = 5, showsd = TRUE, 
              stratisfied = TRUE, print_every_n = 10, early_stopping_rounds = 20, verbose = 1, 
              maximise = FALSE)

xgb <- xgb.train(xgb_params, data = xgb.train, nrounds = bst$best_iteration, 
                 watchlist = list(test = xgb.test, train = xgb.train), print_every_n = 10,
                 early_stop_round = 10, maximise = FALSE, eval_metric = "rmse")
xgb.pred <- predict(xgb, xgb.test)
rmse(xgb.pred, label.test)

xgb.output <- as.data.frame(cbind(price_doc = label.test, PredPrice = xgb.pred))
ggplot(xgb.output) +
  geom_point(aes(x = price_doc, y = price_doc)) +
  geom_point(aes(x = price_doc, y = PredPrice), colour = "red") +
  ggtitle("XGBoost")

# mat <- xgb.importance(feature_names = features2, model = xgb)
# xgb.plot.importance(importance_matrix = mat)

# write.csv(mat, "14 features.csv", row.names = FALSE)
# features2 <- mat$Feature

# RandomForest Model
rf <- randomForest(price_doc ~ ., data = training, ntree = 500)
rf.pred <- predict(rf, testing)
rmse(rf.pred, label.test)

rf.output <- as.data.frame(cbind(price_doc = label.test, PredPrice = rf.pred))
ggplot(rf.output) +
  geom_point(aes(x = price_doc, y = price_doc, size = 2)) +
  geom_point(aes(x = price_doc, y = PredPrice), colour = "red") +
  ggtitle("RandomForest")

# Average of models
avg.pred <- (lm.pred + xgb.pred) / 2
rmse(avg.pred, label.test)

avg.output <- as.data.frame(cbind(price_doc = label.test, PredPrice = avg.pred))
ggplot(avg.output) +
  geom_point(aes(x = price_doc, y = price_doc)) +
  geom_point(aes(x = price_doc, y = PredPrice), colour = "red") +
  ggtitle("AVG LM+XGB")

################################################################
# Submission
################################################################
houseDFx <- houseDF2
id <- houseDFx$id
price <- houseDFx$price_doc
houseDFxp <- houseDFx[, -which(names(houseDF) %in% c("id", "price_doc"))]

preproc <- preProcess(houseDFxp, method = c("BoxCox", "center", "scale"))
houseDFxp <- predict(preproc, houseDFxp)
houseDFxp$price_doc <- log1p(houseDFx$price_doc)

training <- houseDFxp[1:28480, c(features2, "price_doc")]
testing <- houseDFxp[28481:36142, c(features2, "price_doc")]

label.train <- training[, "price_doc"]
label.test <- testing[, "price_doc"]

# LM model
set.seed(1234)
lm <- lm(price_doc ~ ., data = training)
lm.pred <- predict(lm, testing)

# RF model
rf <- randomForest(price_doc ~ ., data = training, ntree = 500)
rf.pred <- predict(rf, testing)

# XGB model
training2 <- as.matrix(training[, features2], rownames.force = NA)
testing2 <- as.matrix(testing[, features2], rownames.force = NA)

training2 <- as(training2, "sparseMatrix")
testing2 <- as(testing2, "sparseMatrix")

xgb.train <- xgb.DMatrix(data = training2, label = label.train)
xgb.test <- xgb.DMatrix(data = testing2, label = label.test)

bst <- xgb.cv(params = xgb_params, data = xgb.train, nrounds = 100, nfold = 5, showsd = TRUE, 
              stratisfied = TRUE, print_every_n = 10, early_stopping_rounds = 20, verbose = 1, 
              maximise = FALSE)

xgb <- xgb.train(xgb_params, data = xgb.train, nrounds = bst$best_iteration, 
                 watchlist = list(test = xgb.test, train = xgb.train), print_every_n = 10, 
                 early_stop_round = 10, maximise = FALSE, eval_metric = "rmse")
xgb.pred <- predict(xgb, xgb.test)

# AVG model
avg.pred <- (lm.pred + xgb.pred) / 2

submit.df <- cbind(Id = id[28481:36142], price_doc = expm1(avg.pred))
write.csv(submit.df, "AVG_SUB_20170629_2.csv", row.names = FALSE)

rm(label.test, label.train, macro, macro2, na_count, partition, testing, training, train2, unchangedDF,
   xgb.output, bst, lm, lm.pred, testing2, training2, xgb, xgb_params, xgb.pred, xgb.test, xgb.train,
   rf, rf.pred, avg.pred, lm.output, rf.output, xgb.output, avg.output)
