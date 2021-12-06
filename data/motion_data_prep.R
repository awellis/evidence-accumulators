library(tidyverse)
library(magrittr)

# df <- read_csv("/Users/lukas/Desktop/imagery_data.csv")
df <- read_csv("/Users/lukas/Desktop/imagery_data_7005.csv")

# resp == "left" --> 1
# instruction == "left" --> 1
# condition == "imagery" --> 1
df %<>% 
  select(id, condition, instruction,
         motion_duration, motion,
         response, rt) %>% 
  rename(resp = response) %>% 
  mutate(id = rep(seq(1, length(unique(id))), each=nrow(df) / length(unique(id))),
         condition = ifelse(condition == "imagery", 1, 0),
         instruction = ifelse(instruction == "left", 1, 0),
         resp = ifelse(resp == "left", 1, 0),
         rt = rt / 1000)


write_csv(df, "/Users/lukas/documents/GitHub/evidence-accumulators/data/single_sub_data.csv")
