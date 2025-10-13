---
title: Analyst Agent
emoji: 🦁
colorFrom: yellow
colorTo: orange
sdk: streamlit
sdk_version: "1.24.1"
app_file: app.py
pinned: false
python_version: "3.11"
---



# Project Success Metrics: The Autonomous AI Data Analyst

The primary goal of this project is to create an autonomous agent that can reliably replicate the workflow of a junior data analyst, transforming a raw dataset into a strategic business report with minimal human intervention.

North Star Metric: Autonomous Insight Generation Rate

This is the single most important metric, combining reliability and quality.

Definition: The percentage of runs that successfully complete end-to-end (from data input to final report) and produce a strategic insight that is both data-grounded and logically sound, without requiring any manual correction or intervention.

How to Measure:

Run the workflow on 10 new, unseen datasets.

Count the number of runs that complete without a terminal error.

A human expert reviews the final reports from the completed runs. The report passes if its core insight is directly supported by the data and the strategic plan is a logical extension of that insight.

Target: > 80%

Tier 1: Core Performance Metrics

These metrics break down the North Star metric into its core components.

1. Reliability & Robustness

Workflow Completion Rate:

Definition: The percentage of workflows that run to the __end__ state without a fatal, unrecoverable error.

How to Measure: (Number of Successful Runs) / (Total Runs Attempted)

Target: > 95% (This allows for occasional, unrecoverable external API failures).

Self-Correction Rate:

Definition: The agent's ability to recover from its own tool-use errors (e.g., coding mistakes, bad API calls) within the retry loop. This is a direct measure of its "intelligence."

How to Measure: (Number of runs with tool errors that still completed successfully) / (Total number of runs that encountered tool errors).

Target: > 90%

2. Quality & Accuracy of Output

Data Grounding Score:

Definition: A measure of how factually accurate the final report is. It ensures the agent is not hallucinating.

How to Measure: Programmatically or manually cross-reference 3-5 key quantitative claims from the generated report (e.g., "South region sales were $389,151") against the actual data by running a separate verification script. The score is the percentage of verified claims.

Target: 100% (The report's quantitative claims must be factually correct).

Plan Execution Fidelity:

Definition: Measures the system's internal consistency. Does the Cleaner_node correctly execute the plan laid out by the PreprocessingPlanner_node?

How to Measure: For each run, compare the Analysis plan against the actions performed in the Cleaner_node log and validated by the Validation_node. This is a pass/fail check for each run.

Target: 100%

Tier 2: Efficiency & Value Metrics

These metrics define the business case for using this system.

1. Efficiency & Cost

Average Cost Per Analysis:

Definition: The total LLM token cost for a single, end-to-end workflow run.

How to Measure: Sum of (Input Tokens * Price) + (Output Tokens * Price).

Target: **< 
0.10
𝑝
𝑒
𝑟
𝑟
𝑢
𝑛
∗
∗
(
𝑌
𝑜
𝑢
𝑟
𝑐
𝑢
𝑟
𝑟
𝑒
𝑛
𝑡
𝑐
𝑜
𝑠
𝑡
𝑜
𝑓
 
0.10perrun∗∗(Yourcurrentcostof 
0.03 is exceptionally good and well below this target).

Average Speed-to-Insight:

Definition: The total wall-clock time from initiating the workflow to the generation of the final Report state.

How to Measure: Time elapsed from main.py start to __end__ state.

Target: < 5 minutes per run (This is a huge advantage over human analysis time).

2. Business Value & Return on Investment (ROI)

Comparative Cost Savings:

Definition: The cost of the AI workflow compared to the cost of a human performing the same task.

How to Measure: (Estimated Human Analyst Cost) vs. (AI Agent Cost).

Human Cost: 2 hours at a junior analyst rate of 
25
/
ℎ
𝑟
=
∗
∗
25/hr=∗∗
50.00**

AI Cost: ~$0.03

Result: A >99.9% cost reduction per analysis.

Comparative Time Savings:

Definition: The time required for the AI workflow compared to a human.

How to Measure: (Estimated Human Analyst Time) vs. (AI Agent Time).

Human Time: 2-4 hours

AI Time: < 5 minutes

Result: A ~98% reduction in time-to-insight.

By framing your project's success with these metrics, you clearly and professionally demonstrate its value not just as a technical curiosity, but as a robust, efficient, and highly valuable business tool.