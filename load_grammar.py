from ratsql import grammars
import json
import csv

def write_products(product_types):
    product_strings = set()
    for product in product_types:
        string = product + " ->"
        print(product)
        fields = product_types[product].fields
        for i, field in enumerate(fields):
            field_type = field.type if not field.seq else field.type+"*"
            field_name = field.name
            string += " " + field_type+"/"+field_name
            if i < len(fields)-1:
                string += ","
        product_strings.add(string)

    return list(product_strings)

def write_sums(sum_types):
    sum_strings = []
    for sum in sum_types:
        for t in sum_types[sum].types:
            string = sum + " -> " + t.name+"/NULL"
            sum_strings.append(string)

    return sum_strings

def write_constructors(constructors):
    constructor_strings = []
    for constructor in constructors:
        string = constructor + " ->"
        fields = constructors[constructor].fields
        for i, field in enumerate(fields):
            string += " " + field.type + "/" + field.name
            if i < len(fields)-1:
                string += ","
        constructor_strings.append(string)

    return constructor_strings

grammar = grammars.spider.SpiderLanguage(output_from = True,
                                         use_table_pointer = True,
                                         include_literals = False,
                                         include_columns = True,
                                         end_with_from = True,
                                         clause_order = None,
                                         infer_from_conditions = True,
                                         factorize_sketch = 0)

to_save = {"product_types": write_products(grammar.ast_wrapper.product_types),
           "sum_types": write_sums(grammar.ast_wrapper.sum_types),
           "primitive_types": [key for key in grammar.ast_wrapper.primitive_types],
           "constructors": write_constructors(grammar.ast_wrapper.constructors)}

# product_strings = write_products(to_save["product_types"])
# sum_strings = write_sums(grammar.ast_wrapper.sum_types)
# constructor_strings = write_constructors(grammar.ast_wrapper.constructors)

# json.dump(to_save, open("ratsql/grammars/spider-0-ast-wrapper.json", 'w'))

with open("ratsql/grammars/spider-0-rules-with-agg-table.csv", 'w') as csvfile:
    writer = csv.writer(csvfile)
    for key in to_save:
        for rule in to_save[key]:
            writer.writerow([key, rule])