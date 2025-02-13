class Partnership:
    def __init__(self, people_instance):
        self.people_instance = people_instance

    def _status_for_age_bracket(self, age_band: str, partnership_status: str):
        occupants = []
        if partnership_status == 'partnered':
            occupants.extend(self._find_and_set_partner(age_band, 'f'))
        else:
            occupants.extend(self._assign_individual_status(age_band))
        print(self.ctxt.size_households)
        print(
            f"Women after {partnership_status} process: {sum(len(lst) for age, lst in self.ctxt.women_by_age.items() if age >= 65)}")
        print(
            f"Men after {partnership_status} process: {sum(len(lst) for age, lst in self.ctxt.men_by_age.items() if age >= 65)}")

        return occupants

    def _assign_individual_status(self, age_band):
        low_age, high_age = map(int, age_band.split('-'))

        # Define possible statuses
        statuses = ['divorced_separated', 'never_partnered', 'widowed']
        all_people_assigned = {'m': [], 'f': []}
        all_individuals = []
        # Loop through both sexes
        for sex in ['f', 'm']:
            # Retrieve the dictionary for the given sex
            people_dict = self.ctxt.women_by_age if sex == 'f' else self.ctxt.men_by_age

            # Loop through each age within the given age range
            for age in filter(lambda x: low_age <= x <= high_age, people_dict.keys()):
                # Extract all people at this age
                people_at_age = people_dict[age]

                # Extract raw probabilities for each status for the given sex and age
                raw_probabilities = [
                    self.ctxt.partnerships_lad.loc[(sex, status), age] for status in statuses
                ]

                normalized_probabilities = raw_probabilities / sum(raw_probabilities)

                # Assign status to each individual at this age
                for person in people_at_age:
                    assigned_status = np.random.choice(statuses, p=normalized_probabilities)
                    person.partnership_status = assigned_status
                    all_people_assigned[sex].append(person)
                    all_individuals.append([person])

            # print("___________")
            # print(f"People of sex {sex}:")
            # for status in statuses:
            #     print(f"Status: {status}")
            #     people_in_status = [person for person in all_people_assigned[sex] if
            #                         person.partnership_status == status]
            #     for i, person in enumerate(people_in_status):
            #         print(f'{i + 1}: ID: {person.id}, Age: {person.age}, Partnership Status: {person.partnership_status}, Spouse: {person.spouse}')
            #     print('...............')

        for sex_to_remove, (people_dict, people_to_remove) in {
            'm': [self.ctxt.men_by_age, all_people_assigned['m']],
            'f': [self.ctxt.women_by_age, all_people_assigned['f']]
        }.items():
            self._remove_from_people_dict(people_dict, people_to_remove)

        return all_individuals

    def _find_and_set_partner(self, age_band, sex):
        low_age, high_age = age_band.split('-')
        low_age, high_age = int(low_age), int(high_age)

        num_to_assign = self.ctxt.partnerships_oa.loc[(sex, 'partnered'), age_band]
        if num_to_assign == 0: return
        # print(f"Assigning: {num_to_assign} partners")

        eligible_women = []
        probabilities = []
        for age in filter(lambda x: low_age <= x, self.ctxt.women_by_age.keys()):
            prob = self.ctxt.partnerships_lad.loc[(sex, 'partnered'), age]
            women_at_age = self.ctxt.women_by_age[age]
            eligible_women.extend(women_at_age)
            probabilities.extend([prob] * len(women_at_age))
        if not eligible_women:
            raise Exception("NO ELIGIBLE WOMEN IN AGE RANGE")

        probabilities = probabilities / sum(probabilities)
        chosen_women_indices = np.random.choice(range(len(eligible_women)), size=num_to_assign, replace=False,
                                                p=probabilities)
        chosen_women = [eligible_women[i] for i in chosen_women_indices]

        # print("___________")
        # print("Chosen women:")
        successfully_matched = {'f': [], 'm': []}
        couples = []
        for i, person in enumerate(chosen_women):
            self._get_hetero_partner(person) if self._get_sexuality(person) == 1 else self._get_same_sex_partner()
            spouse = person.spouse
            # print(f'{i+1}: ID: {person.id}, Sex: {person.sex}, Age: {person.age}, Partnership Status: {person.partnership_status}, Spouse ID: {person.spouse.id}')
            # print(f'Spouse: ID: {spouse.id}, Sex: {spouse.sex}, Age: {spouse.age}, Partnership Status: {spouse.partnership_status}, Spouse ID: {spouse.spouse.id}')
            successfully_matched[person.sex].append(person)
            successfully_matched[spouse.sex].append(spouse)
            couples.append([person, spouse])

        for sex_to_remove, (people_dict, people_to_remove) in {
            'm': [self.ctxt.men_by_age, successfully_matched['m']],
            'f': [self.ctxt.women_by_age, successfully_matched['f']]
        }.items():
            self._remove_from_people_dict(people_dict, people_to_remove)

        return couples

    def _get_random_person_in_age_bracket(self):
        pass

    def _get_sexuality(self, person):
        # TODO: Add logic for determining sexuality... Placeholder is all Heterosexual (1)
        return 1

    def _get_hetero_partner(self, person):  # ToDo For cleanliness, I suggest we integrate same sex here too.
        age, sex = person.age, person.sex
        prob_df = self.distributor.couples_age_disparity['hetero']
        prob_age = prob_df[age]
        prob_age = prob_age[prob_age != 0]
        valid_ages = prob_age.index

        spouse_options = self.ctxt.men_by_age if sex == 'f' else self.ctxt.women_by_age

        eligible_spouses = []
        spouse_probabilities = []
        for i, age in enumerate(filter(lambda x: x in valid_ages, self.ctxt.men_by_age.keys())):
            eligible_spouses_at_age = spouse_options[age]
            eligible_spouses.extend(eligible_spouses_at_age)
            spouse_probabilities.extend([prob_age.loc[age]] * len(eligible_spouses_at_age))
        if not eligible_spouses:
            print("NO ELIGIBLE SPOUSES IN AGE RANGE")
            exit()

        spouse_probabilities = spouse_probabilities / sum(spouse_probabilities)
        chosen_spouse = np.random.choice(eligible_spouses, size=1, replace=False, p=spouse_probabilities)[0]
        person.spouse, person.partnership_status = chosen_spouse, 'partnered'
        chosen_spouse.spouse, chosen_spouse.partnership_status = person, 'partnered'

        return chosen_spouse

    def _get_same_sex_partner(self):
        spouse = None  # ToDo
        return spouse