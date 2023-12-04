<template>
  <div id="app">
    <b-card>
      <b-card-header>
        <input type="file" ref="file" name="inputName" v-on:change="fileData(this)"/>
        <button @click="submit" class="d-inline ml-3">Submit</button>
      </b-card-header>
      <b-card-body>
        <b>{{output}}</b>
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
      output: ''
    }
  },
  methods: {
    fileData() {
      this.file = this.$refs.file.files[0];
      console.log(this.file)
    },
    submit() {
      let formData = new FormData()
      formData.append('file', this.file)
      this.output = "Loading ..."

      axios.post("http://127.0.0.1:5000/audio_endpoint",
          formData,
          {
            headers: {
              'Accept': 'application/json',
              'Content-Type': 'multipart/form-data'
            }
          }).then((data) => {
            console.log(data)
            this.output = JSON.stringify(data.request.body)
      })
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
