<template>
  <div>

    <h3>Liste d'appartements</h3>
    <v-row>
      <v-col cols="12" md="8">
        <v-data-table :headers="headers" :items="suites" class="elevation-1">
          <template v-slot:item.nbRooms="{ item }">
            <span>{{ item.nbRooms }}</span>
          </template>
          <template v-slot:item.surface="{ item }">
            <span>{{ item.surface }} m²</span>
          </template>
          <template v-slot:item.nbWindows="{ item }">
            <span>{{ item.nbWindows }}</span>
          </template>
          <!-- <template v-slot:item.year="{ item }">
            <span>{{ item.year }}</span>
        </template>
        <template v-slot:item.balcony="{ item }">
            <span>{{ item.balcony }}</span>
        </template>
        <template v-slot:item.garage="{ item }">
            <span>{{ item.garage }}</span>
        </template>
        <template v-slot:item.city="{ item }">
            <span>{{ item.city }}</span>
        </template>
        <template v-slot:item.price_category="{ item }">
            <span>{{ item.price_category }}</span>
        </template>
        <template v-slot:item.price="{ item }">
            <span>{{ item.price }}</span>
        </template> -->
          <template v-slot:item.actions="{ item }">
            <v-btn color="red" @click="supprimerAppartement(item.id)">
              Supprimer
            </v-btn>
          </template>
        </v-data-table>
      </v-col>

      <v-col cols="12" md="4">
        <add @appartement-ajoute="ajouterAppartement" />
      </v-col>
    </v-row>
  </div>


</template>

<script>
import { defineComponent, reactive, onMounted } from 'vue';
import Add from './Add.vue';
import axios from 'axios';

export default defineComponent({
  components: {
    Add
  },
  setup() {
    const headers = [
      { text: 'Nombre de chambres', value: 'nbRooms' },
      { text: 'Surface (m²)', value: 'surface' },
      { text: 'Nombre de fenêtres', value: 'nbWindows' },
      // { text: 'Année', value: 'year' },
      // { text: 'Balcon', value: 'balcony' },
      // { text: 'Garage', value: 'garage' },
      // { text: 'Ville', value: 'city' },
      // { text: 'Catégorie de prix', value: 'price_category' },
      // { text: 'Prix', value: 'price' },
      { text: 'Actions', value: 'actions', sortable: false }
    ];

    let suites = reactive([]);

    const fetchAppartements = async () => {
      try {
        const response = await axios.get('http://localhost:5000/suites')
          .then((response) => {
            return response.data;
          });
        suites.push(...response);
      } catch (error) {
        console.error('Erreur lors du chargement des appartements :', error);
      }
    };

    onMounted(fetchAppartements);

    const ajouterAppartement = async (nouvelAppartement) => {
      const response = await axios.post('http://localhost:5000/suites', { id: suites.length + 1, ...nouvelAppartement })
        .then((response) => {
          return response.data;
        });

      suites = response
      suites.push({ ...nouvelAppartement, id: suites.length + 1 });
    };

    const supprimerAppartement = (id) => {
      const index = suites.findIndex((suite) => suite.id === id);
      if (index !== -1) {
        suites.splice(index, 1);
      }
    };


    return { headers, suites, ajouterAppartement, supprimerAppartement };
  }
});
</script>