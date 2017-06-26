
def pretty_print_list(d, indent=0):
    for value in d:
        pretty_print(value, indent+1)

def pretty_print_dict(d, indent=0):
    for key, value in d.iteritems():
        print '\t' * indent + str(key)
        pretty_print(value, indent+1)

def pretty_print(value, indent=0):
    if isinstance(value, dict):
        pretty_print_dict(value, indent + 1)
    elif isinstance(value, list):
        print "array:["
        pretty_print_list(value, indent + 1)
        print "]"
    else:
        print '\t' * (indent+1) + str(value)
