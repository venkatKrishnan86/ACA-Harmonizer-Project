<template>
  <div id="app">
    <h2>Chordinator</h2>
    <b-card>
      <b-card-header>
        <input type="file" ref="file" name="inputName" v-on:change="fileData(this)"/>
        <button @click="submit" class="d-inline ml-3">Submit</button>
      </b-card-header>
      <b-card-body>
        <b-badge v-if="loading" variant="danger"> loading...</b-badge>
        <audio controls v-if="audio_src" :src="audio_src"/>
        <template v-if="chords && timestamps">
          <table class="table table-striped">
            <thead>
              <tr>
                <th>Chord</th>
                <th>Time</th>
              </tr>
            </thead>
              <tr v-for="(chord, index) in chords" :key="index">
                <td>{{chord}}</td>
                <td>{{timestamps[index]}}</td>
              </tr>
          </table>
        </template>
      </b-card-body>
    </b-card>

  </div>
</template>

<script>

import axios from "axios";

export default {
  name: 'App',
  data() {
    return {
      file: null,
      loading: false,
      audio_src: null,
      chords: null,
      timestamps: null
    }
  },
  methods: {
    fileData() {
      this.file = this.$refs.file.files[0];
      console.log(this.file)
    },
    async submit() {
      let formData = new FormData()
      formData.append('file', this.file)
      this.loading = true

      const timestamp_data = await axios.post("http://127.0.0.1:5000/run_model",
          formData,
          {
            headers: {
              'Accept': 'application/json',
              'Content-Type': 'multipart/form-data'
            }
          })

      this.chords = timestamp_data.data[0]
      this.timestamps = timestamp_data.data[1]

      const audio_data = await axios.get("http://127.0.0.1:5000/fetch_cached_audio")
      const wav = new Blob([audio_data.data], {type: 'audio/wav'})
      this.audio_src = (window.URL || window.webkitURL).createObjectURL(wav)
      this.loading = false
    }
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
