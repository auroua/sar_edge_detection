class Person(object):
    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name

    @property
    def full_name(self):
        return '%s  %s' %(self.first_name, self.last_name)

    @full_name.setter
    def full_name(self, value):
        self.first_name = value

    def get_first_name(self):
        return self.first_name

    def set_first_name(self, value):
        self.first_name = value

    name = property(get_first_name, set_first_name)


if __name__=='__main__':
    person = Person('Mike', 'Driscoll')
    print person.full_name

    person.full_name = 'test property'
    print person.full_name
    # person.set_first_name('java')
    # print person.full_name
    #
    # person.name = 'python'
    # print person.full_name