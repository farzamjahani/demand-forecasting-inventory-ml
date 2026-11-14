-- Run against data/retail_demo.sqlite with sqlite3 or a SQLite extension.

-- 1. Revenue and units sold by product in the last 30 days of available sales.
SELECT
    p.product_name,
    SUM(s.quantity) AS units_sold,
    ROUND(SUM(s.quantity * p.unit_price), 2) AS estimated_revenue
FROM daily_sales AS s
JOIN products AS p ON p.product_id = s.product_id
WHERE date(s.date) > date((SELECT MAX(date) FROM daily_sales), '-30 days')
GROUP BY p.product_id, p.product_name
ORDER BY estimated_revenue DESC;

-- 2. Number of sales days and average daily demand by category.
SELECT p.category,
       ROUND(AVG(s.quantity), 2) AS avg_daily_units_per_product,
       COUNT(*) AS observed_product_days
FROM daily_sales AS s
JOIN products AS p ON p.product_id = s.product_id
GROUP BY p.category
ORDER BY avg_daily_units_per_product DESC;

-- 3. Products requiring a replenishment order based on the simple reorder point.
SELECT product_name, stock_on_hand, reorder_point, suggested_order_qty
FROM replenishment
WHERE suggested_order_qty > 0
ORDER BY suggested_order_qty DESC;

-- 4. 7-day rolling units sold, by product (SQL window function).
SELECT date, product_id, quantity,
       ROUND(AVG(quantity) OVER (
         PARTITION BY product_id ORDER BY date
         ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
       ), 2) AS trailing_7_day_average
FROM daily_sales
WHERE product_id = 101
ORDER BY date DESC
LIMIT 14;

-- 5. CTE: sales value by product and category.
WITH product_revenue AS (
    SELECT p.product_id, p.category, p.product_name,
           SUM(s.quantity * p.unit_price) AS revenue
    FROM products AS p
    JOIN daily_sales AS s ON s.product_id = p.product_id
    GROUP BY p.product_id, p.category, p.product_name
)
SELECT category, ROUND(SUM(revenue), 2) AS category_revenue
FROM product_revenue
GROUP BY category
ORDER BY category_revenue DESC;

-- 6. Lead-time impact on replenishment for selected SKU at 95% nominal service.
SELECT product_name, lead_time_days, target_service_level,
       safety_stock, reorder_point, suggested_order_qty
FROM lead_time_scenarios
WHERE product_id = 101 AND target_service_level = '95%'
ORDER BY lead_time_days;

-- 7. Highest projected shortfall for 14-day lead time and 95% service.
SELECT product_name, lead_time_days, suggested_order_qty
FROM lead_time_scenarios
WHERE lead_time_days = 14 AND target_service_level = '95%'
ORDER BY suggested_order_qty DESC;
