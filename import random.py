import random
# a coins tossing simulation
coin_trials = 10000
heads = 0
tails = 0

for _ in range(coin_trials):
    toss = random.choice(['H', 'T'])
    if toss == 'H':
        heads += 1
    else:
        tails += 1
        
print(f"coins toss simulation ({coin_trials} trials):")
print(f"probability of heads: {heads / coin_trials:.4f}")
print(f"probability of tails: {tails / coin_trials:.4f}")
   
# b dice rolling simulation
dice_trials = 10000
sum7 = 0
for _ in range(dice_trials):
    die1 = random.randint(1, 6)
    die2 = random.randint(1, 6)
    if die1 + die2 == 7:
        sum7 += 1

print(f"dice rolling simulation ({dice_trials} trials):")
print(f"probability of sum7: {sum7 / dice_trials:.4f}")

# c probability of at least one 6 in ten rolls of a dice
def prob_at_least_one_six(trials=10000, rolls=10):
    count = 0
    for _ in range(trials):
        if 6 in [random.randint(1, 6) for _ in range(rolls)]:
            count += 1
    return count / trials

probability = prob_at_least_one_six()
print(f"probability of at least one 6 in 10 rolls: {probability:.4f}")

#d conditional probability and bayes theorem with colorful balls
#draw from a bag which have 5 red, 7 green and 8 blue balls, draw it 1000 times with replacement.
# [1] estimate probability (red ball/previous blue ball).
# [2] verify bayes theorem using results.

#[1] estimate probability (red ball/previous blue ball).
import random 

colors = ["red"]*5 + ["green"]*7 + ["blue"]*8
draws = 1000
sequence = [random.choice(colors) for _ in range(draws)]
count_prev_blue = 0
count_red_given_blue = 0

for i in range(1, len(sequence)):
    if sequence[i-1] == "blue":
        count_prev_blue += 1
    if sequence[i] == "red" and sequence[i-1] == "blue":
        count_red_given_blue += 1

if count_prev_blue > 0:
    prob_red_given_blue = count_red_given_blue / count_prev_blue
    print(f"Conditional probability P(red | previous blue): {prob_red_given_blue}")
else:
    prob_red_given_blue = 0
    print(f"Conditional probability P(red | previous blue): {prob_red_given_blue}")

#[2] verify bayes theorem using results.
total_red = sequence.count("red")
total_blue = sequence.count("blue")
P_red = total_red/ draws
P_blue = total_blue/draws

count_prev_red = 0
count_blue_given_red = 0
for i in range(1, len(sequence)):
    if sequence [i-1] == "red":
        count_prev_red += 1
        if sequence[i] == "blue":
            count_blue_given_red += 1

            if count_prev_red > 0:
                prob_blue_given_red = count_blue_given_red/count_prev_red
            else:
                prob_blue_given_red = 0
                #by bayes theorem P(red/blue) = [P(blue/red)*{P(red)/P(blue)}]
bayes_estimate = (prob_blue_given_red*P_red)/ P_blue if P_blue > 0 else 0

print(f"bayes estimate prob(red/blue): {bayes_estimate:.4f}")


#Sample 1000 values from a variable X with P(X=1)=0.25, P(X=2)=0.35, P(X=3)=0.4, then compute mean, variance, std.
import numpy as np

values = [1,2,3]
probs = [0.25,0.35,0.4]
sample = np.random.choice(values, size = 1000, p=probs)

mean = np.mean(sample)
variance = np.var(sample)
std_dev = np.std(sample)

print(f"Discrete RV sample stats (n=1000)")
print(f"Empirical Mean: {mean:>4f}")
print(f"Empirical Variance: {variance:.4f}")
print(f"Empirical Std_dev: {std_dev:.4f}")


#Generate 2000 samples from an exponential distribution (mean=5), plot histogram and PDF overlay.
import numpy as np
import matplotlib.pyplot as plt

exp_samples = np.random.exponential(scale=5, size=2000)

plt.hist(exp_samples, bins=30, density=True, alpha=0.6, color="skyblue", label="histogram")

from scipy.stats import expon
X = np.linspace(0,np.max(exp_samples),200)
pdf = expon.pdf(X, scale=5)
plt.plot(X,pdf,"r--",label="PDF (mean=5)")
plt.title("Exponential Distribution (meand=5)")
plt.xlabel("value")
plt.ylabel("Destity")
plt.legend()
plt.show()

    #• Generate 10,000 uniform random numbers.
    #• Take 1000 samples of size 30, compute sample means.
    #• Plot the original uniform distribution and the distribution of means.
uniform_data = np.random.uniform(low=0, high=1, size=10000)
sample_means = [np.mean(np.random.choice(uniform_data, size=30, replace=False)) for _ in range(1000) ]
plt.figure(figsize=(12,5))
plt.hist(uniform_data, bins=30, color="lightgreen", alpha=0.7)
plt.title("Uniform Distribution (original data)")
plt.subplot(1,2,2)
plt.hist(sample_means, bins=30, color="orange", alpha=0.7)
plt.title("Sample means Distribution (n=30, 1000 samples)")
plt.show()