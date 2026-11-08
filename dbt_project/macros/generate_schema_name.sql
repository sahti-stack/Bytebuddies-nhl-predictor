{#
    This macro ensures that custom schemas (like 'bronze', 'silver', 'gold')
    are used directly instead of being prefixed with the default schema (e.g. 'main_silver').
#}
{% macro generate_schema_name(custom_schema_name, node) -%}

    {%- set default_schema = target.schema -%}
    {%- if custom_schema_name is none -%}

        {{ default_schema }}

    {%- else -%}

        {{ custom_schema_name | trim }}

    {%- endif -%}

{%- endmacro %}
