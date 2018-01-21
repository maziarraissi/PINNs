---
layout: default
---


We introduce physics informed neural networks -- neural networks that are trained to solve supervised learning tasks while respecting any given law of physics described by general nonlinear [partial differential equations](https://en.wikipedia.org/wiki/Partial_differential_equation). We present our developments in the context of solving two main classes of problems: [data-driven solution](https://arxiv.org/abs/1711.10561) and [data-driven discovery](https://arxiv.org/abs/1711.10566) of partial differential equations. Depending on the nature and arrangement of the available data, we devise two distinct classes of algorithms, namely continuous time and discrete time models. The resulting neural networks form a new class of data-efficient universal function approximators that naturally encode any underlying physical laws as prior information. In the first part, we demonstrate how these networks can be used to [infer solutions to partial differential equations](https://arxiv.org/abs/1703.10230), and obtain physics-informed surrogate models that are fully differentiable with respect to all input coordinates and free parameters. In the second part, we focus on the problem of [data-driven discovery of partial differential equations](https://arxiv.org/abs/1708.00588).

* * * * * *
### Data-driven Solutions of Nonlinear Partial Differential Equations


In this [first part](https://arxiv.org/abs/1711.10561) of our two-part treatise, we focus on computing data-driven solutions to partial differential equations of the general form

$$
u_t + \mathcal{N}[u] = 0,\ x \in \Omega, \ t\in[0,T],
$$

where $$u(t,x)$$ denotes the latent (hidden) solution, $$\mathcal{N}[\cdot]$$ is a nonlinear differential operator, and $$\Omega$$ is a subset of $$\mathbb{R}^D$$. In what follows, we put forth two distinct classes of algorithms, namely continuous and discrete time models, and highlight their properties and performance through the lens of different benchmark problems. All code and data-sets are available [here](https://github.com/maziarraissi/PINNs).


Text can be **bold**, _italic_, or ~~strikethrough~~.

[Link to another page](another-page).

There should be whitespace between paragraphs.

There should be whitespace between paragraphs. We recommend including a README, or a file with information about your project.

# [](#header-1)Header 1

This is a normal paragraph following a header. GitHub is a code hosting platform for version control and collaboration. It lets you and others work together on projects from anywhere.

## [](#header-2)Header 2

> This is a blockquote following a header.
>
> When something is important enough, you do it even if the odds are not in your favor.

### [](#header-3)Header 3

```js
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```

```ruby
# Ruby code with syntax highlighting
GitHubPages::Dependencies.gems.each do |gem, version|
  s.add_dependency(gem, "= #{version}")
end
```

#### [](#header-4)Header 4

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.

##### [](#header-5)Header 5

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

###### [](#header-6)Header 6

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

### There's a horizontal rule below this.

* * *

### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip

### And an ordered list:

1.  Item one
1.  Item two
1.  Item three
1.  Item four

### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item

### Small image

![](https://assets-cdn.github.com/images/icons/emoji/octocat.png)

### Large image

![](https://guides.github.com/activities/hello-world/branching.png)


### Definition lists can be used with HTML syntax.

<dl>
<dt>Name</dt>
<dd>Godzilla</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl>

```
Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this.
```

```
The final element.
```

