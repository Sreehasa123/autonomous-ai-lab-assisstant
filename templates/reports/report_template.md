# Autonomous AI Lab Assistant: {{ experiment.name }}

**Experiment ID:** `{{ experiment.id }}`  
**Status:** `{{ experiment.status }}`  
**Version:** `v{{ experiment.current_version }}`  
**Date:** {{ timestamp }}  

---

## 1. Problem Formulation
> {{ experiment.problem_statement }}

## 2. Dataset Overview
- **Name:** {{ dataset.name }}
- **Provider:** {{ dataset.source_provider }}
- **Target Column:** `{{ dataset.target_column }}`
- **Dimensions:** {{ dataset.shape }}

## 3. Preprocessing & Modeling
- **Architecture:** `{{ model.model_name }}` ({{ model.framework }})
- **Scaling:** {{ preprocessing.scale_numeric }}
- **Missing Values:** {{ preprocessing.handle_missing }}

## 4. Empirical Evaluation
{% for metric_name, metric_val in metrics.items() %}
- **{{ metric_name }}:** `{{ metric_val }}`
{% endfor %}

## 5. Next Steps
{% for rec in recommendations %}
- **{{ rec.title }}:** {{ rec.action }}
{% endfor %}
