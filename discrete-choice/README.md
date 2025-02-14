# Discrete Choice models

Videos:
- [Discrete choice, part 1: Conditional logit and multinomial logit - Anders Munk-Nielsen](https://youtu.be/8Hm2GCKdd5g?si=ZKG8XteB49V0lB5p)
- [Discrete choice, part 2: The independence of irrelevant alternatives (IIA) - Anders Munk-Nielsen](https://youtu.be/1Eg6TvOx2OY?si=jzeAZHQicOW4_GMR)

Notebooks:
- Conditional Logit (CL) toy example
- Multi-nomial Logit (MNL) toy example 
- Instrument Variable (IV) toy example

## For Maersk

### **Defining the Conditional Logit (CL) Problem for Maersk Spot**
To successfully **model customer choices**, estimate **price elasticities**, and **optimize revenue**, we need to structure the **Conditional Logit (CL) problem** effectively. Below is a structured approach:

---

## **1. Market Definition**
### **Decision Context**
- Customers book container shipping **without a contract** via *maersk.com*.
- At the time of booking (2-6 weeks before departure), they are presented with multiple **alternative sailings** (varying in departure time, origin, destination, route, and sometimes container type).
- Customers choose **one option** among **a varying number of alternatives**.

### **Choice Set**
Each customer's **choice set** consists of:
- **Departure time** (e.g., different weeks or days)
- **Origin-destination pair** (e.g., choosing a nearby port instead)
- **Route** (direct vs. transshipment)
- **Container size/type** (e.g., 20ft, 40ft, reefers)
- **Additional services** (e.g., store-door pickup/delivery)

The **main challenge** is that the **number of alternatives is dynamic**, and many **combinations receive zero bookings**.

---

## **2. Key Alternative-Specific Characteristics**
Since we are initially **excluding customer characteristics**, we focus on **alternative-only attributes** that influence choice.

### **Primary Attributes to Include**
| Feature | Description | Hypothesis |
|---------|------------|------------|
| **Price (€/$)** | Cost of the shipment | Customers are price-sensitive; price elasticity varies across segments |
| **Departure time** | Time until departure (e.g., 2 weeks, 4 weeks) | Customers value earlier departures but may trade off price |
| **Transit time** | Total transport duration (port-to-port or door-to-door) | Shorter transit times are preferred |
| **Alternative origin** | Nearby origin ports available | Some customers are flexible on origin |
| **Alternative destination** | Nearby destination ports available | Some customers are flexible on destination |
| **Route type** | Direct vs. transshipment route | Direct is preferred but may depend on price |
| **Capacity availability** | Whether the alternative is likely to sell out soon | Scarcity may drive urgency and bookings |
| **Container size/type** | Standard 20ft/40ft, reefers, etc. | Cross-elasticity between sizes, especially for certain commodities |
| **Add-ons** | Store-door pickup/delivery option | Customers might value convenience at a price |

---

## **3. Data Gathering & Structuring the Choice Data**
To estimate the **Conditional Logit Model**, we need **structured choice data**. 

### **Data Required per Customer Search**
For every search event:
1. **Observed choice** → The alternative the customer actually books (if any).
2. **Choice set** → The full list of **available alternatives** shown at the time of booking.
3. **Alternative attributes** → Price, departure time, transit time, origin, destination, etc.
4. **Availability status** → Whether an alternative sold out before booking.

### **Dealing with Zero-Bookings**
Since many product combinations **never get booked**, we should:
- **Log all choice sets**, not just the chosen one.
- Include **“no purchase” as a choice**, meaning some customers may leave without booking.
- Consider **aggregating rare choices** if needed (e.g., grouping routes).

---

## **4. Model Estimation**
### **Utility Specification**
The standard **utility function** in the **Conditional Logit Model** is:

$$
U_{ij} = \beta_1 \cdot \text{Price}_{ij} + \beta_2 \cdot \text{DepartureTime}_{ij} + \beta_3 \cdot \text{TransitTime}_{ij} + \beta_4 \cdot \text{AlternativeOrigin}_{ij} + \beta_5 \cdot \text{AlternativeDestination}_{ij} + \beta_6 \cdot \text{RouteType}_{ij} + \beta_7 \cdot \text{CapacityAvailability}_{ij} + \beta_8 \cdot \text{ContainerType}_{ij} + \beta_9 \cdot \text{AddOns}_{ij} + \epsilon_{ij}
$$

Where:
- $ U_{ij} $ = Utility of customer $ i $ choosing alternative $ j $.
- The **$\beta$ coefficients** are estimated using Maximum Likelihood Estimation (MLE).
- The **$\epsilon_{ij}$** term represents random unobserved factors.

### **Estimation Approach**
1. **Fit a Conditional Logit Model (Baseline)** using **only alternative-specific attributes**.
2. **Test Nested Logit or Mixed Logit models** to account for correlation between similar alternatives (e.g., close departure weeks).
3. **Validate predictions** using out-of-sample choice data.

---

## **5. Price Optimization for Revenue Maximization**
Once the model estimates **price elasticities**, we can **optimize pricing** by:

### **A. Simulating Price Sensitivity**
- Use estimated **price coefficients ($\beta_1$)** to model **demand response** to price changes.
- Forecast **how bookings will shift** if we raise/lower prices.

### **B. Revenue Maximization Strategy**
We optimize **prices dynamically** to maximize expected revenue:

$$
\max_{p_j} \sum_j P_j(p_j) \cdot p_j
$$

where:
- $ P_j(p_j) $ is the predicted probability of booking option $ j $ at price $ p_j $.
- $ p_j $ is the price of alternative $ j $.

### **C. Implementation**
1. **Set a baseline pricing strategy** (e.g., cost-based or historical price levels).
2. **Run simulations** to test how different price levels affect revenue.
3. **Implement real-time price updates** based on demand fluctuations.

---

## **Next Steps**
✅ **Define market**: Customers choosing shipping options on Maersk Spot.  
✅ **Set up data gathering**: Log all choice sets & bookings.  
✅ **Run estimations**: Estimate price elasticities and choice probabilities.  
🔜 **Optimize pricing**: Use the estimated model to maximize revenue.

## Market-level Conditional Logit (CL) model

Yes, you **can model market shares** instead of individual customer choices using a **market-level Conditional Logit (CL) model**. This is useful when **individual-level data is unavailable** or when you want to model **aggregate demand and price optimization** at a broader level.

---

## **1. How Market Share Modeling Works**
Instead of modeling **individual choices**, you estimate the **share of total demand** for each alternative in a given market (e.g., shipping route, time period, or customer segment). 

### **Key Differences from Individual Choice Models**
| Aspect | Individual-Level CL | Market-Share CL |
|--------|---------------------|-----------------|
| **Data Unit** | Each customer's choice | Aggregate demand for alternatives |
| **Dependent Variable** | Chosen alternative (1/0) | Market share (continuous proportion) |
| **Choice Set** | Varies per individual | Fixed within a market segment |
| **Objective** | Predict individual behavior | Predict total demand for each option |

---

## **2. Market Share Logit Model Formulation**
The **market share of alternative \( j \)** is modeled using a **logit function**:

\[
S_j = \frac{e^{U_j}}{\sum_k e^{U_k}}
\]

where:
- \( S_j \) = **Market share of alternative \( j \)** (fraction of total demand).
- \( U_j \) = **Systematic utility** of alternative \( j \), usually modeled as:

\[
U_j = \beta_1 \cdot \text{Price}_j + \beta_2 \cdot \text{TransitTime}_j + \beta_3 \cdot \text{DepartureWeek}_j + \beta_4 \cdot \text{RouteType}_j + \epsilon_j
\]

- The market share equation is **analogous to individual choice probabilities**, but instead of modeling **binary choices**, we model **continuous shares**.

---

## **3. Data Setup for Market Share Estimation**
Each row in the dataset represents a **shipping alternative** in a given market segment (e.g., week, region, or route).

| Week | Origin | Destination | Price ($) | Transit Time (days) | Market Share (%) |
|------|--------|------------|-----------|----------------------|------------------|
| 1    | Shanghai | Rotterdam | 1500 | 30 | 40% |
| 1    | Shanghai | Rotterdam | 1400 | 32 | 35% |
| 1    | Shanghai | Rotterdam | 1300 | 34 | 25% |
| 2    | Shanghai | Rotterdam | 1600 | 29 | 50% |
| 2    | Shanghai | Rotterdam | 1500 | 31 | 30% |

---

## **4. Estimating the Market Share Logit Model**
Since we are modeling **continuous shares**, we use **Maximum Likelihood Estimation (MLE)** to fit the model.

### **Python Implementation**
Here's how to estimate a **market share logit model**:

```python
import numpy as np
import pandas as pd
import scipy.optimize as opt

# Simulated market share dataset
data = pd.DataFrame({
    "Price": [1500, 1400, 1300, 1600, 1500],
    "TransitTime": [30, 32, 34, 29, 31],
    "MarketShare": [0.40, 0.35, 0.25, 0.50, 0.30]
})

# Utility function: U_j = β1 * Price + β2 * TransitTime
def utility(beta, X):
    return np.dot(X, beta)

# Log-likelihood function for market shares
def log_likelihood(beta, X, shares):
    U = utility(beta, X)
    exp_U = np.exp(U - np.max(U))  # Prevent numerical instability
    P = exp_U / np.sum(exp_U)  # Market share prediction
    return -np.sum(shares * np.log(P))  # Negative log-likelihood

# Prepare data for estimation
X = data[["Price", "TransitTime"]].values  # Independent variables
shares = data["MarketShare"].values  # Observed market shares
initial_beta = np.zeros(X.shape[1])  # Initial guess for coefficients

# Estimate parameters using MLE
result = opt.minimize(log_likelihood, initial_beta, args=(X, shares), method='BFGS')
print("Estimated Coefficients:", result.x)
```

---

## **5. Optimizing Prices for Revenue Maximization**
Once we estimate the **price sensitivity coefficient (\(\beta_1\))**, we can **optimize pricing** by adjusting prices to **maximize total revenue**:

\[
\max_{p_j} \sum_j S_j(p_j) \cdot p_j
\]

### **Revenue Optimization Process**
1. **Use the estimated model** to forecast how market shares change with price.
2. **Optimize prices** using `scipy.optimize` to maximize:

   \[
   \text{Revenue} = \sum_j S_j(p_j) \cdot p_j
   \]

3. **Adjust prices dynamically** to respond to demand shifts.

### **Python Code for Revenue Optimization**
```python
# Revenue function
def revenue_function(prices):
    U_opt = result.x[0] * prices + result.x[1] * data["TransitTime"].values
    exp_U = np.exp(U_opt - np.max(U_opt))
    S_opt = exp_U / np.sum(exp_U)  # Predicted market shares
    revenue = np.sum(S_opt * prices)  # Total revenue
    return -revenue  # Minimize negative revenue to maximize revenue

# Optimize prices
optimal_prices = opt.minimize(revenue_function, data["Price"].values, method='Nelder-Mead').x
print("Optimal Prices:", optimal_prices)
```

---

## **6. Key Benefits of Market Share Modeling**
✅ **Scales Better**: No need for individual-level data; works well for aggregated demand.  
✅ **Optimizes Prices Dynamically**: Uses estimated elasticities to adjust prices in real-time.  
✅ **Accounts for Cross-Elasticity**: Can model how demand shifts between alternatives (e.g., different departure weeks or routes).  

Would you like to refine this further for **specific market segments** (e.g., different origin-destination pairs)? 🚀

## Aggregate Products (Penny Groups)

Aggregating products into **pre-existing product groups** can be a powerful way to improve **estimation stability** and **market share modeling**, especially when dealing with many product combinations that receive **zero bookings**. However, it also introduces **trade-offs** between accuracy and interpretability.

---

## **1. Why Aggregate Products?**
🚀 **Advantages of Aggregation:**
- **Reduces sparsity** → Many individual products have zero sales, making estimation difficult.
- **Improves model stability** → More data points per alternative improve estimation precision.
- **Enables market share modeling** → We can define **group-level market shares** instead of noisy individual bookings.
- **Captures meaningful substitution effects** → Aggregated groups reflect real substitution patterns.

⚠️ **Challenges:**
- **Loses fine-grained insights** → Aggregation hides differentiation within a group.
- **Risk of grouping unrelated products** → Poorly chosen groups may distort estimation.
- **Cross-elasticity estimation is harder** → If cross-group substitution exists, it needs to be modeled separately.

---

## **2. How to Define Product Groups for Aggregation?**
Since some products are **never sold**, we should **group alternatives meaningfully** based on:
1. **Shared attributes** (e.g., same origin-destination, departure week range).
2. **Observed substitution behavior** (e.g., customers often switch between options in the same group).
3. **Commercial definitions** (e.g., Maersk’s existing product categories).

### **Example of Aggregated Product Groups**
Instead of modeling **individual product choices**, we model **group-level choices**:

| Product Group | Price ($) | Transit Time (days) | Market Share (%) |
|--------------|----------|---------------------|------------------|
| Short-haul, Early Departure | 1400 | 25 | 35% |
| Short-haul, Late Departure | 1300 | 30 | 25% |
| Long-haul, Early Departure | 1600 | 40 | 20% |
| Long-haul, Late Departure | 1500 | 45 | 20% |

Here, instead of tracking **each individual shipping option**, we **aggregate choices into product groups** based on:
- **Route type** (short-haul vs. long-haul).
- **Departure timing** (early vs. late).

We then model **group-level market shares** using a **market share logit model**.

---

## **3. How to Modify the Market Share Model for Aggregated Products?**
Once we define product groups, we estimate a **market share logit model** where each row represents a **product group, not an individual product**.

### **Updated Python Code for Aggregated Products**
```python
import numpy as np
import pandas as pd
import scipy.optimize as opt

# Simulated aggregated product group data
data = pd.DataFrame({
    "ProductGroup": ["Short-haul Early", "Short-haul Late", "Long-haul Early", "Long-haul Late"],
    "Price": [1400, 1300, 1600, 1500],
    "TransitTime": [25, 30, 40, 45],
    "MarketShare": [0.35, 0.25, 0.20, 0.20]  # Aggregated market share per group
})

# Utility function: U_j = β1 * Price + β2 * TransitTime
def utility(beta, X):
    return np.dot(X, beta)

# Log-likelihood function for aggregated market shares
def log_likelihood(beta, X, shares):
    U = utility(beta, X)
    exp_U = np.exp(U - np.max(U))  # Prevent overflow
    P = exp_U / np.sum(exp_U)  # Market share prediction
    return -np.sum(shares * np.log(P))  # Negative log-likelihood

# Prepare data for estimation
X = data[["Price", "TransitTime"]].values  # Independent variables
shares = data["MarketShare"].values  # Observed market shares
initial_beta = np.zeros(X.shape[1])  # Initial guess for coefficients

# Estimate parameters using MLE
result = opt.minimize(log_likelihood, initial_beta, args=(X, shares), method='BFGS')
print("Estimated Coefficients:", result.x)
```

---

## **4. Optimizing Prices for Aggregated Product Groups**
Once we estimate price sensitivities, we can **optimize group-level prices** for revenue maximization:

### **Updated Revenue Optimization**
```python
# Revenue function for aggregated product groups
def revenue_function(prices):
    U_opt = result.x[0] * prices + result.x[1] * data["TransitTime"].values
    exp_U = np.exp(U_opt - np.max(U_opt))
    S_opt = exp_U / np.sum(exp_U)  # Predicted market shares
    revenue = np.sum(S_opt * prices)  # Total revenue
    return -revenue  # Minimize negative revenue to maximize revenue

# Optimize prices across product groups
optimal_prices = opt.minimize(revenue_function, data["Price"].values, method='Nelder-Mead').x
print("Optimal Prices:", optimal_prices)
```

---

## **5. When Is Aggregation Useful?**
✅ **Market-Level Strategy:** If optimizing **market share & revenue** rather than individual bookings.  
✅ **High Zero-Booking Rate:** When many individual products have **no demand**, aggregation helps.  
✅ **Simplifying Decision-Making:** Helps management adjust pricing **at a higher level** (e.g., by route type).  

🚨 **When to Avoid Aggregation?**
- If customers have **strong preferences for specific alternatives**.
- If differentiation within groups is **important for pricing**.
- If cross-elasticity within a group is **highly variable**.

---

## **Final Recommendation**
🔹 If you need **stable estimates**, aggregate products into well-defined groups.  
🔹 If you need **granular price optimization**, consider **nested logit models** to account for within-group substitution.  
🔹 **A hybrid approach** may work: estimate demand at the **group level** but optimize prices **within groups** dynamically.

Would you like to explore **cross-elasticities between groups** to refine price sensitivity? 🚀
